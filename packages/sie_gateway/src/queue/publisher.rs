use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_nats::jetstream;
use dashmap::{DashMap, DashSet};
use futures_util::future::try_join_all;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, oneshot};
use tracing::{debug, info, warn};
use utoipa::ToSchema;

use rmp::decode::read_str_from_slice;
use rmp::Marker;

use super::payload_store::PayloadStore;
use super::streaming::{
    ChunkApplied, ChunkEnvelope, ChunkError, NakEnvelope, StreamCollector, StreamOutcome,
};
use crate::endpoint::InferenceEndpoint;
use crate::metrics;

const PAYLOAD_OFFLOAD_THRESHOLD: usize = 1_024 * 1_024; // 1 MB

// H9 — first-chunk-fallback rate limit defaults.
//
// The gateway republishes generation work to the pool subject when the
// direct-dispatched worker doesn't send a first chunk within the
// first-chunk window. Without bounds an in-cluster outage (cold workers,
// model-load storm) trips the fallback for every request in flight,
// doubling the JetStream pressure on the pool stream and amplifying the
// load that already caused the timeouts. The token bucket caps the
// fallback rate per (model, pool); requests beyond the burst are refused
// with a 504 so the client can decide whether to retry.
//
// 5 / sec sustained with a burst of 10 covers a healthy cluster's
// occasional cold-start fallbacks without rate-limiting them, while
// preventing a runaway storm from consuming the pool stream's pending
// budget. Values are tunable per deployment via the constructor in a
// future iteration; the constants here are the "safe default" the audit
// recommended.
const FALLBACK_RATE_PER_SEC_DEFAULT: f64 = 5.0;
const FALLBACK_BURST_DEFAULT: f64 = 10.0;

/// Simple monotonic-time token bucket. Not thread-safe on its own; the
/// caller wraps a single instance in the appropriate concurrency
/// primitive (the publisher uses ``DashMap<key, Mutex<TokenBucket>>``).
///
/// ``try_take`` returns true and decrements the available tokens by one
/// when at least one whole token is available; otherwise it returns
/// false without mutating state. Tokens accrue continuously at
/// ``rate_per_sec`` up to ``burst``.
#[derive(Debug)]
struct TokenBucket {
    /// Currently available tokens (fractional). May briefly exceed
    /// ``burst`` if the system clock is set backwards — capped on the
    /// next refill.
    tokens: f64,
    /// Steady-state refill rate.
    rate_per_sec: f64,
    /// Maximum tokens. ``rate_per_sec * window + 1`` is the upper bound
    /// on bursts the bucket will permit before refusing.
    burst: f64,
    /// Last accrual computation timestamp.
    last_refill: Instant,
}

impl TokenBucket {
    fn new(rate_per_sec: f64, burst: f64) -> Self {
        Self {
            tokens: burst,
            rate_per_sec,
            burst,
            last_refill: Instant::now(),
        }
    }

    fn try_take(&mut self) -> bool {
        let now = Instant::now();
        let elapsed = now
            .saturating_duration_since(self.last_refill)
            .as_secs_f64();
        self.last_refill = now;
        self.tokens = (self.tokens + elapsed * self.rate_per_sec).min(self.burst);
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

/// Parameters extracted from the request body for top-level WorkItem fields.
///
/// `query_item` is stored as an `rmpv::Value` rather than
/// `serde_json::Value` so that msgpack-in requests (the hot path for
/// clients that send binary payloads) can pass the decoded value
/// straight through to the worker without an intermediate
/// JSON-shaped round-trip. For JSON-in requests the body is
/// converted once via [`json_to_rmpv`] before being stored here.
///
/// Small configuration fields (`options`, `output_schema`) stay as
/// `serde_json::Value` — they never carry binary data, and keeping
/// them JSON-shaped avoids rewriting the (de)serializer for the
/// small config surface.
#[derive(Debug, Clone, Default)]
pub struct WorkParams {
    pub output_types: Option<Vec<String>>,
    pub instruction: Option<String>,
    pub is_query: bool,
    pub options: Option<serde_json::Value>,
    pub labels: Option<Vec<String>>,
    pub output_schema: Option<serde_json::Value>,
    pub query_item: Option<rmpv::Value>,
    /// Generate-only typed params (walking-skeleton wire shape). Mutually exclusive
    /// with the encode/score/extract fields above; populated only when
    /// ``endpoint == "generate"``.
    pub generate: Option<GenerateParams>,
    /// Routing-affinity hint. Carried onto the work
    /// envelope and made visible to routing logic without
    /// unpacking ``generate``. Currently read-but-ignored by the worker.
    pub routing_key: Option<String>,
    /// Prompt-cache hint. Same semantics as
    /// :attr:`routing_key`.
    pub prompt_cache_key: Option<String>,
}

/// Discriminated input for a generate work item.
///
/// The original wire shape was ``{prompt: String, ...}`` flat under
/// ``GenerateParams``. The chat-completions surface introduces an
/// OpenAI-shaped ``{messages: [...], ...}`` variant; the two are
/// mutually exclusive on a single work item. Both shapes still
/// serialise / deserialise as flat keys under :class:`GenerateParams`
/// (``#[serde(untagged)]`` + ``#[serde(flatten)]`` on the enclosing
/// field) so a prompt-only worker receiving a chat-shaped published
/// prompt item is wire-compatible.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum GenerateInput {
    /// Raw prompt string — prompt wire shape.
    Prompt { prompt: String },
    /// OpenAI-shaped chat messages. The worker renders the
    /// chat template with the model's tokenizer before forwarding to
    /// the generation adapter.
    Messages { messages: Vec<ChatMessage> },
}

impl Default for GenerateInput {
    fn default() -> Self {
        GenerateInput::Prompt {
            prompt: String::new(),
        }
    }
}

/// One image attached to a chat message, extracted from an OpenAI
/// ``image_url`` data URI at the gateway.
///
/// ``data`` is the **base64 payload string**, NOT raw bytes. This is
/// deliberate: the generate work item travels through the sidecar's
/// ``WorkItem.generate: serde_json::Value``, and ``serde_json::Value``
/// cannot hold a msgpack ``bin`` (rmp_serde rejects it with "invalid type:
/// byte array"). A base64 string round-trips cleanly through
/// ``serde_json::Value`` with no sidecar changes. The worker base64-decodes
/// it back to bytes (see ``_parse_message_images_field``).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatImage {
    /// Base64-encoded image bytes (standard alphabet, no ``data:`` prefix).
    pub data: String,
    /// Format hint parsed from the data-URI MIME subtype (e.g. ``"png"``).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
}

/// One ordered content part of a multimodal chat message — either a text
/// fragment or an image placeholder (the bytes ride the message's flat
/// ``images`` list, consumed in order as ``Image`` parts are encountered).
/// Serializes internally-tagged so the worker reads ``{"type":"text",…}`` /
/// ``{"type":"image"}`` and round-trips through ``serde_json::Value`` on the
/// sidecar (no msgpack ``bin`` — same constraint as :struct:`ChatImage`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    Image,
}

/// One chat message in the OpenAI request shape. Role is validated
/// against the allowed set at the gateway; ``content`` carries the text
/// (multi-part text parts are concatenated). Any ``image_url`` parts are
/// decoded into ``images`` and gated on the model's vision capability
/// after model resolution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    /// Assistant tool-call requests (OpenAI shape), preserved across a
    /// multi-turn tool exchange so the worker can replay them into the
    /// chat template. ``None`` on plain messages.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
    /// Links a ``role:"tool"`` result message back to the assistant
    /// tool-call it answers. Required by OpenAI on tool messages.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Vision input: images decoded from the message's ``image_url`` /
    /// ``input_image`` content parts. ``None`` on text-only messages. The
    /// worker renders one placeholder per image, in order, and forwards the
    /// bytes to the engine.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<ChatImage>>,
    /// Ordered content layout for multimodal messages: the original
    /// text↔image interleaving from the OpenAI ``content`` parts array.
    /// ``None`` on text-only / non-interleaved-legacy messages — the worker
    /// then falls back to the images-first ``content`` + ``images`` render.
    /// When present, the worker emits one image placeholder per ``Image`` part
    /// *in place*, pulling bytes from ``images`` in order; only placeholder
    /// positions change (the flat ``images`` list / ``image_data`` to the
    /// engine is unchanged).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_parts: Option<Vec<ContentPart>>,
}

/// Structured-output grammar.
///
/// Wire shape mirrors the Python ``sie_server.types.grammar.GrammarSpec``
/// dataclass. Serde discriminates on ``kind`` (``"json_schema"`` |
/// ``"regex"``) and carries the schema-or-pattern payload under
/// ``value`` plus optional ``label`` / ``strict`` from the OpenAI
/// ``response_format.json_schema`` wrapper.
///
/// The two variants are mutually exclusive on a single request; the
/// gateway's :func:`handlers::grammar::parse_grammar` is the only place
/// that builds these and enforces all safety caps (payload size, schema
/// depth, regex length, JSON-Schema reject-list) before the worker sees
/// the grammar.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GrammarSpec {
    JsonSchema {
        value: serde_json::Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        label: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        strict: Option<bool>,
    },
    Regex {
        value: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        label: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        strict: Option<bool>,
    },
    /// EBNF / context-free grammar. Forwarded to the worker which
    /// dispatches to Outlines or XGrammar (both support EBNF natively).
    /// Subject to :const:`MAX_GRAMMAR_BYTES` at the gateway; no
    /// further structural walk is performed (the gateway does not
    /// parse EBNF — the worker's backend is the authority).
    Ebnf {
        value: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        label: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        strict: Option<bool>,
    },
}

/// Generation parameters carried verbatim from the HTTP request body
/// through the work envelope to the worker's ``StreamingProcessor``.
///
/// ``input`` flattens into the parent map so the on-the-wire shape stays
/// ``{prompt | messages, max_new_tokens, ...}`` — backwards-compatible
/// with the original streaming work items.
///
/// `routing_key` / `prompt_cache_key` are caller-supplied
/// affinity hints used by the gateway for HRW direct-dispatch (xxh3
/// hash → per-worker subject). The raw strings are forwarded to the
/// worker so it can use them for cache lookups; the gateway hashes them
/// before any logging.
///
/// **Privacy contract:** `safety_identifier` is intentionally absent.
/// The HTTP layer parses it and discards it without logging the value.
/// Adding it here would put potentially-PII strings on the JetStream wire.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GenerateParams {
    #[serde(flatten)]
    pub input: GenerateInput,
    pub max_new_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    /// OpenAI ``frequency_penalty``: range validated upstream to
    /// ``[-2.0, 2.0]``. Forwarded verbatim to the worker; absent → the
    /// worker uses the sampler default (typically 0.0).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    /// OpenAI ``presence_penalty``: same shape and validation as
    /// :attr:`frequency_penalty`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    /// Non-OpenAI ``top_k`` (Together / Fireworks / vLLM extension):
    /// integer ``>= 1``, gateway-validated. Forwarded to SGLang's
    /// ``sampling_params["top_k"]``. Absent → top-k disabled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Non-OpenAI ``repetition_penalty``: float in ``(0.0, 2.0]``,
    /// gateway-validated. Forwarded to SGLang's
    /// ``sampling_params["repetition_penalty"]``. Absent → sampler default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f64>,
    /// SGLang ``sampling_params["min_new_tokens"]``: integer ``>= 0``,
    /// gateway-validated. Minimum tokens the model must emit before
    /// any stop condition can fire. Absent → sampler default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_tokens: Option<u32>,
    /// Per-request kwargs forwarded to the tokenizer's
    /// ``apply_chat_template``. Worker merges them on top of the model
    /// YAML defaults (YAML wins on conflict). Validated as a JSON
    /// object by the gateway. Absent → only YAML defaults apply.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_template_kwargs: Option<serde_json::Value>,
    /// Structured-output spec. Absent when the request omitted the
    /// ``grammar`` field (SIE-native) or ``response_format`` (OpenAI
    /// chat). Populated by :func:`handlers::grammar::parse_grammar`
    /// after all gateway-side safety caps have passed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grammar: Option<GrammarSpec>,
    /// Caller-supplied routing affinity hint. Highest-priority input
    /// to the HRW key resolution in `crate::routing::key`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routing_key: Option<String>,
    /// OpenAI-compatible cache-key hint. Second-priority input to HRW
    /// key resolution; also passed verbatim so the worker can use it
    /// for adapter-level cache lookups.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    /// OpenAI ``tools``: non-empty array of ``{type: "function",
    /// function: {name, description?, parameters?}}``. Forwarded
    /// verbatim to the worker; the worker uses presence (rather than
    /// the schemas themselves) to enable the
    /// ``parse_tool_call_stream`` pipeline. The gateway has run the
    /// JSON-Schema safety walker on each ``function.parameters``.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<serde_json::Value>>,
    /// OpenAI ``tool_choice``: one of ``"auto"`` / ``"none"`` /
    /// ``"required"`` or ``{type:"function", function:{name}}``.
    /// Informational on the worker today (Qwen3's chat template emits
    /// ``<tool_call>`` blocks based on the model's own decision); we
    /// still plumb it so future sampler-level constraints can read it
    /// without a wire-shape bump.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    /// OpenAI ``parallel_tool_calls`` (default ``true``). Currently
    /// informational; surfaced on the envelope so worker-side
    /// observability can label requests by parallelism intent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    /// OpenAI ``seed`` — sampler determinism hint. Best-effort:
    /// kernel non-determinism, batching order, and KV-cache reuse all
    /// defeat exact reproducibility. Forwarded to the worker which
    /// sets it on the SGLang ``sampling_params``. Absent → SGLang's
    /// own (non-deterministic) default. Gateway-validated as a u64.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// OpenAI ``logit_bias`` — per-token additive bias on the
    /// sampler. Keys are token-id strings (OpenAI's wire shape), values
    /// are floats in ``[-100.0, 100.0]``. Gateway clamps the map size
    /// and per-value range; the worker forwards verbatim to SGLang's
    /// ``logit_bias`` sampling-param. Absent → unbiased.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<std::collections::BTreeMap<String, f64>>,
    /// OpenAI ``logprobs`` flag — when ``true`` the worker requests
    /// per-token log-probabilities from SGLang and surfaces them on
    /// each generation chunk. Absent or ``false`` → no logprobs in
    /// the response (``choices[i].logprobs: null``).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    /// OpenAI ``top_logprobs`` — how many alternative tokens to
    /// surface per position. Range ``[0, 20]`` per OpenAI's spec;
    /// gateway clamps. Requires ``logprobs: true``.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,
    /// OpenAI ``n`` — number of candidate generations. ``1`` is the
    /// gateway default. ``n>1`` is supported only on non-streaming
    /// requests; the chat handler returns 400 for ``n>1 && stream:true``.
    /// Forwarded to the worker which sets SGLang's ``sampling_params.n``
    /// and surfaces ``n`` outputs in a single ``WorkResult``.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    /// OpenAI ``best_of``: generate this many candidates server-side and return
    /// the top ``n`` ranked by cumulative logprob. Must satisfy ``best_of >= n``.
    /// Non-streaming only. ``None`` → behaves as ``best_of == n``.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub best_of: Option<u32>,
    /// Whether the client requested SSE streaming. The worker only needs this
    /// for ``n>1``: streaming fans the candidates out as per-``choice_index``
    /// delta chunks, vs. the single terminal ``candidates[]`` array used for the
    /// non-streaming aggregate. ``false``/absent → unchanged single-candidate
    /// behaviour.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub stream: bool,
    /// Multi-LoRA: the public served-name of a LoRA adapter to apply (declared
    /// in the model profile's ``lora_paths``). ``None`` → the base model.
    /// Forwarded to the worker, which passes it as SGLang's
    /// ``sampling_params.lora_path`` (in-batch per-request adapter selection).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lora_adapter: Option<String>,
}

/// Borrowed, serialize-only view of the worker work-item envelope.
///
/// Used on the publish hot path to avoid cloning per-item the fields
/// that every item in a batch shares (pool/model/gpu/router_id/
/// reply_subject/operation/bundle_config_hash and the whole
/// `WorkParams` block including `options` / `output_schema` which can
/// be non-trivial JSON trees). For an N-item encode/extract request
/// this saves roughly `7N` small-string clones plus `4N` deep
/// `Option<Vec<_>> / Option<serde_json::Value>` clones.
///
/// The field names, order, and serde attributes are guarded by tests
/// against an owned envelope shape so `rmp_serde::to_vec_named(&WorkItemRef)`
/// keeps producing the msgpack map the Python worker consumes.
/// Deserialization is intentionally not supported; results arrive as
/// `WorkResult`.
#[derive(Debug, Serialize)]
struct WorkItemRef<'a> {
    pub work_item_id: &'a str,
    pub request_id: &'a str,
    pub item_index: u32,
    pub total_items: u32,
    pub operation: &'a str,
    pub model_id: &'a str,
    pub profile_id: &'a str,
    pub engine: &'a str,
    pub pool_name: &'a str,
    #[serde(skip_serializing_if = "str::is_empty")]
    pub admission_pool: &'a str,
    pub machine_profile: &'a str,
    pub item: Option<&'a rmpv::Value>,
    pub payload_ref: Option<&'a str>,
    pub output_types: Option<&'a [String]>,
    pub instruction: Option<&'a str>,
    pub is_query: bool,
    pub options: Option<&'a serde_json::Value>,
    pub query_item: Option<&'a rmpv::Value>,
    pub query_payload_ref: Option<&'a str>,
    pub score_items: Option<&'a [rmpv::Value]>,
    pub labels: Option<&'a [String]>,
    pub output_schema: Option<&'a serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generate: Option<&'a GenerateParams>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub routing_key: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<&'a str>,
    pub bundle_config_hash: &'a str,
    pub router_id: &'a str,
    pub reply_subject: &'a str,
    pub timestamp: f64,
    /// W3C Trace Context. Skipped on serialisation when `None`,
    /// preserving the compact pre-observability wire shape when no
    /// trace context is present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub traceparent: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tracestate: Option<&'a str>,
}

/// Per-request context that every [`WorkItemRef`] in a batch borrows
/// from. Grouping these lets us build the per-item view with a single
/// struct literal and keeps `publish_single` / `publish_score` from
/// taking a dozen `&str` arguments each.
struct WorkItemShared<'a> {
    request_id: &'a str,
    endpoint: &'a str,
    model: &'a str,
    pool: &'a str,
    admission_pool: &'a str,
    gpu: &'a str,
    engine: &'a str,
    bundle_config_hash: &'a str,
    router_id: &'a str,
    reply_subject: &'a str,
    params: &'a WorkParams,
    timestamp: f64,
    /// W3C trace context captured once at publish-call time. The
    /// shared block is what every per-item [`WorkItemRef`] borrows
    /// from, so the propagator runs once per request rather than
    /// once per fan-out item.
    traceparent: Option<&'a str>,
    tracestate: Option<&'a str>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkResult {
    #[serde(default)]
    pub work_item_id: String,
    pub request_id: String,
    #[serde(default)]
    pub item_index: u32,
    #[serde(default)]
    pub success: bool,
    #[serde(default, with = "serde_bytes")]
    pub result_msgpack: Vec<u8>,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub error_code: Option<String>,
    #[serde(default)]
    pub inference_ms: Option<f64>,
    #[serde(default)]
    pub queue_ms: Option<f64>,
    #[serde(default)]
    pub processing_ms: Option<f64>,
    #[serde(default)]
    pub worker_id: Option<String>,
    #[serde(default)]
    pub tokenization_ms: Option<f64>,
    #[serde(default)]
    pub postprocessing_ms: Option<f64>,
    #[serde(default)]
    pub payload_fetch_ms: Option<f64>,
}

struct CachedStreamInfo {
    num_pending: u64,
    num_consumers: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
pub struct PendingGenerationGroup {
    pub model: String,
    pub display_model: String,
    pub pool: String,
    pub count: u64,
    pub waiting_first_chunk: u64,
    pub active_streams: u64,
    pub republished: u64,
    pub oldest_request_age_ms: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
pub struct PendingGenerationSnapshot {
    pub total: u64,
    pub groups: Vec<PendingGenerationGroup>,
}

impl PendingGenerationSnapshot {
    pub fn for_model(&self, model_id: &str) -> Self {
        let groups: Vec<PendingGenerationGroup> = self
            .groups
            .iter()
            .filter(|group| group.display_model == model_id || group.model == model_id)
            .cloned()
            .collect();
        Self {
            total: groups.iter().map(|group| group.count).sum(),
            groups,
        }
    }
}

fn accumulate_pending_generation_group(
    grouped: &mut BTreeMap<(String, String, String), PendingGenerationGroup>,
    collector: &StreamCollector,
    now: Instant,
) {
    let key = (
        collector.model.clone(),
        collector.display_model.clone(),
        collector.pool.clone(),
    );
    let age_ms = now
        .saturating_duration_since(collector.published_at)
        .as_millis()
        .min(u128::from(u64::MAX)) as u64;

    let group = grouped
        .entry(key.clone())
        .or_insert_with(|| PendingGenerationGroup {
            model: key.0,
            display_model: key.1,
            pool: key.2,
            count: 0,
            waiting_first_chunk: 0,
            active_streams: 0,
            republished: 0,
            oldest_request_age_ms: 0,
        });
    group.count += 1;
    if collector.first_chunk_at.is_some() {
        group.active_streams += 1;
    } else {
        group.waiting_first_chunk += 1;
    }
    if collector.republished {
        group.republished += 1;
    }
    group.oldest_request_age_ms = group.oldest_request_age_ms.max(age_ms);
}

pub struct WorkPublisher {
    jetstream: jetstream::Context,
    router_id: String,
    payload_store: Arc<dyn PayloadStore>,
    result_timeout: Duration,
    max_stream_pending: u64,
    stream_max_age: Duration,
    /// Pools we've already `get_or_create_stream`'d for, keyed by pool
    /// name so we skip the admin-API round trip on subsequent
    /// requests. The value is the pre-computed JetStream stream name
    /// (`WORK_POOL_{pool}`) so the publish hot path doesn't rebuild
    /// it with `format!` on every call.
    ensured_streams: DashMap<String, Arc<str>>,
    /// Backpressure snapshot, keyed by pool. The background monitor
    /// refreshes this every tick; the first request to a cold pool
    /// primes it synchronously in [`Self::ensure_stream`] so we
    /// don't fail-open during the initial monitor interval.
    stream_info_cache: DashMap<String, CachedStreamInfo>,
    pending_results: DashMap<String, ResultCollector>,
    /// Streaming aggregator. Keyed on ``request_id`` like
    /// ``pending_results`` but holds the per-chunk state for one
    /// generation request. Populated by ``publish_generate_streaming``
    /// and drained by ``handle_inbox`` when chunk envelopes arrive.
    pending_streams: DashMap<String, StreamCollector>,
    /// Request ids whose generate work item was offloaded to the object
    /// store (``payload_ref`` set, ``generate`` blanked). The blob lives at
    /// ``{request_id}_0.bin`` and must be deleted once the stream finishes —
    /// promptly at the terminal funnel (``handle_chunk``) and, as a backstop
    /// for failure/timeout terminations, reconciled against ``pending_streams``
    /// by the periodic ``cleanup_expired`` sweep.
    offloaded_streams: DashSet<String>,
    /// Object-store payload keys written or attempted for non-streaming
    /// requests. Keyed by request id so cleanup deletes only concrete keys
    /// instead of deriving every possible ``{id}_{i}.bin`` / ``{id}_score.bin``
    /// name from ``total_items``. Failed deletes stay tracked so
    /// ``cleanup_expired`` can retry them after the request collector is gone.
    offloaded_payload_keys: DashMap<String, BTreeSet<String>>,
    /// Core NATS client (cancel publishes use core NATS, not
    /// JetStream). Populated lazily by
    /// :meth:`start_inbox_subscription`; reads on the cancel path
    /// tolerate ``None`` (cancellation simply becomes a no-op).
    nats_client: tokio::sync::RwLock<Option<async_nats::Client>>,
    inbox_handle: tokio::sync::Mutex<Option<tokio::task::JoinHandle<()>>>,
    /// H9 — first-chunk-fallback rate-limit buckets, keyed by
    /// ``"{model}|{pool}"``. Each bucket gates calls into
    /// [`Self::republish_to_pool`] tagged with the
    /// ``first_chunk_timeout`` reason. The NAK-driven republish path
    /// bypasses the bucket because that path is already throttled by
    /// the worker's own NAK rate. Wrapped in ``std::sync::Mutex``
    /// because the critical section is purely CPU-bound (refill +
    /// compare) and we don't hold it across awaits.
    fallback_buckets: DashMap<String, std::sync::Mutex<TokenBucket>>,
    fallback_rate_per_sec: f64,
    fallback_burst: f64,
}

struct ResultCollector {
    _total_items: u32,
    results: Vec<Option<WorkResult>>,
    sender: Option<oneshot::Sender<Vec<WorkResult>>>,
    deadline: Instant,
    operation: String,
    published_at: Instant,
    pool_label: String,
    model_label: String,
    pool_fallback_subject: Option<String>,
    direct_fallback_payloads: Vec<Option<Vec<u8>>>,
    direct_fallback_republished: bool,
}

struct PublishedWorkItem {
    index: usize,
    ack: jetstream::context::PublishAckFuture,
    encoded: Vec<u8>,
}

type DirectFallbackPayload = (usize, Vec<u8>);
type DirectFallbackPayloads = Vec<DirectFallbackPayload>;

fn take_direct_fallback_payloads(
    collector: &mut ResultCollector,
) -> Option<(String, DirectFallbackPayloads)> {
    if collector.direct_fallback_republished {
        return None;
    }
    let subject = collector.pool_fallback_subject.clone()?;
    let mut payloads = Vec::new();
    for (index, payload) in collector.direct_fallback_payloads.iter().enumerate() {
        let still_missing = collector.results.get(index).is_some_and(Option::is_none);
        if still_missing {
            if let Some(bytes) = payload.clone() {
                payloads.push((index, bytes));
            }
        }
    }
    if payloads.is_empty() {
        return None;
    }
    collector.direct_fallback_republished = true;
    Some((subject, payloads))
}

fn stream_name(pool: &str) -> String {
    format!("WORK_POOL_{}", pool)
}

fn canonical_stream_subjects(observed: Vec<String>, desired: &str) -> Option<Vec<String>> {
    if observed.len() == 1 && observed.first().is_some_and(|subject| subject == desired) {
        None
    } else {
        Some(vec![desired.to_string()])
    }
}

/// Normalize a model ID or operator-provided lane id for use as a single
/// NATS subject token.
///
/// The workers' JetStream pull consumer filters on
/// `sie.work.{pool}.{machine_profile}.{bundle}.*`, which matches
/// **exactly one token** in the model-ID position. NATS subject tokens
/// legally contain `/` but MUST NOT contain `.`, `*`, `>`, or whitespace.
/// Without this normalization, a model with `.` in its id (e.g.
/// `vidore/colqwen2.5-v0.2`) would expand into multiple tokens, the publish
/// would not match the stream's subject filter, and JetStream would reject it
/// (surfacing to the client as a 504 / no-consumer error).
///
/// The mapping must stay in lockstep with the Python SDK
/// (`sie_sdk.queue_types.normalize_model_id`) so that workers and the gateway
/// agree on the wire-level subject:
///
///     `/`     -> `__`
///     `.`     -> `_dot_`
///     `*`     -> `_`
///     `>`     -> `_`
///     ` `     -> `_`
///
/// The encoding is not fully reversible — e.g. `org/a__b` and `org/a/b` both
/// collapse to the same token — but this is safe in practice because
/// HuggingFace model IDs do not contain literal `__`.
fn normalize_model_id(model_id: &str) -> String {
    let mut out = String::with_capacity(model_id.len() + 8);
    for ch in model_id.chars() {
        match ch {
            '/' => out.push_str("__"),
            '.' => out.push_str("_dot_"),
            '*' | '>' | ' ' => out.push('_'),
            // Control chars (`\n`, `\r`, `\t`, …) are whitespace/illegal
            // in a NATS subject token but are not caught by the literal
            // arms above. Map them to `_` too so the empty-registry
            // fallback path (which can interpolate an unsanitized model
            // id) can't emit a malformed subject.
            c if c.is_control() => out.push('_'),
            c => out.push(c),
        }
    }
    out
}

fn work_subject(pool: &str, machine_profile: &str, bundle: &str, model: &str) -> String {
    format!(
        "sie.work.{}.{}.{}.{}",
        pool,
        normalize_model_id(machine_profile),
        normalize_model_id(bundle),
        normalize_model_id(model)
    )
}

/// Per-worker subject `sie.work.{pool}.{machine_profile}.{bundle}.{model}.{worker_id}`.
///
/// Matches `sie_sdk.queue_types.work_worker_subject` so workers and
/// the gateway agree on the wire-level subject. The pool stream
/// filters on `sie.work.{pool}.*.*.*`, so this worker-addressed subject
/// cannot be captured by it — exactly the double-delivery guarantee
/// the design requires.
fn work_subject_worker(
    pool: &str,
    machine_profile: &str,
    bundle: &str,
    model: &str,
    worker_id: &str,
) -> String {
    // Worker ids are operator-controlled (set via `WorkerStatusMessage.name`,
    // ultimately sourced from `SIE_WORKER_ID` / `HOSTNAME` / `POD_NAME` on the
    // worker side) and would otherwise be interpolated verbatim into a NATS
    // subject — an id containing `.`, `*`, `>`, or whitespace would produce
    // an illegal subject and every HRW pick that landed on it would fail
    // with "no responders". Apply the same scrub as the model id so wonky
    // names (notably Kubernetes pod hostnames like
    // `sie-worker-7d9f-default-0.sie-worker.default.svc`) degrade to
    // deterministic underscore-joined tokens instead of disappearing from
    // the cluster.
    //
    // CROSS-LANGUAGE CONTRACT (workstream G-M5): this normalization MUST
    // produce byte-identical output to `sie_sdk.queue_types.normalize_worker_id`
    // in Python (which delegates to `normalize_model_id` for exactly this
    // reason). If you change the mapping here, mirror the change in the
    // Python helper or direct-dispatch will silently miss every worker whose
    // raw id contains the newly-changed character.
    format!(
        "sie.work.{}.{}.{}.{}.{}",
        pool,
        normalize_model_id(machine_profile),
        normalize_model_id(bundle),
        normalize_model_id(model),
        normalize_model_id(worker_id)
    )
}

/// Where a `WorkItem` should be published.
///
/// Direct-dispatch: when the HRW pick yields an
/// eligible worker, the gateway publishes to that worker's per-worker
/// subject. When no worker is eligible (empty ring, key resolution
/// missed, post-NAK / post-timeout republish), it falls back to the
/// pool subject — same target the original walking-skeleton code always used.
#[derive(Debug, Clone)]
pub enum PublishTarget {
    /// Direct-dispatch to a specific worker. Subject:
    /// `sie.work.{pool}.{machine_profile}.{bundle}.{model}.{worker_id}`.
    Worker {
        pool: String,
        machine_profile: String,
        bundle: String,
        model: String,
        worker_id: String,
    },
    /// Pool fan-out — any worker subscribed to
    /// `sie.work.{pool}.{machine_profile}.{bundle}.*` can pick it up.
    Pool {
        pool: String,
        machine_profile: String,
        bundle: String,
        model: String,
    },
}

impl PublishTarget {
    /// Resolve the JetStream subject for this target.
    pub fn subject(&self) -> String {
        match self {
            PublishTarget::Worker {
                pool,
                machine_profile,
                bundle,
                model,
                worker_id,
            } => work_subject_worker(pool, machine_profile, bundle, model, worker_id),
            PublishTarget::Pool {
                pool,
                machine_profile,
                bundle,
                model,
            } => work_subject(pool, machine_profile, bundle, model),
        }
    }

    /// Stable metric label describing the target kind. Wired up to
    /// publish-side metrics in a follow-up; carried in the API
    /// surface now so future callers don't have to extend `PublishTarget`.
    #[allow(dead_code)]
    pub fn label(&self) -> &'static str {
        match self {
            PublishTarget::Worker { .. } => "worker",
            PublishTarget::Pool { .. } => "pool",
        }
    }

    /// Build the `Pool` fallback that corresponds to a `Worker` target.
    /// Carried in the API for admission-control use cases that may need to
    /// materialise the fallback target outside `republish_to_pool`.
    #[allow(dead_code)]
    pub fn as_pool_fallback(&self) -> PublishTarget {
        match self {
            PublishTarget::Worker {
                pool,
                machine_profile,
                bundle,
                model,
                ..
            }
            | PublishTarget::Pool {
                pool,
                machine_profile,
                bundle,
                model,
            } => PublishTarget::Pool {
                pool: pool.clone(),
                machine_profile: machine_profile.clone(),
                bundle: bundle.clone(),
                model: model.clone(),
            },
        }
    }

    pub fn model(&self) -> &str {
        match self {
            PublishTarget::Worker { model, .. } | PublishTarget::Pool { model, .. } => model,
        }
    }

    pub fn pool(&self) -> &str {
        match self {
            PublishTarget::Worker { pool, .. } | PublishTarget::Pool { pool, .. } => pool,
        }
    }

    pub fn machine_profile(&self) -> &str {
        match self {
            PublishTarget::Worker {
                machine_profile, ..
            }
            | PublishTarget::Pool {
                machine_profile, ..
            } => machine_profile,
        }
    }
}

/// Peek the top-level `kind` discriminator on an already-decoded
/// `rmpv::Value`. Returns `Some("chunk")` / `Some("nak")` / `Some(...)`
/// for a map carrying a string `kind`, or `None` for a non-map value
/// (e.g. the array-shaped `WorkResult`) or a map without a string
/// `kind`. Lets `handle_inbox` dispatch on the decoded value without a
/// second decode from the raw slice.
fn envelope_kind(value: &rmpv::Value) -> Option<&str> {
    let rmpv::Value::Map(entries) = value else {
        return None;
    };
    for (k, v) in entries {
        if let rmpv::Value::String(s) = k {
            if s.as_str() == Some("kind") {
                return match v {
                    rmpv::Value::String(s) => s.as_str(),
                    _ => None,
                };
            }
        }
    }
    None
}

/// Fast-path extraction of `request_id` from raw msgpack bytes.
/// Returns None on any parse failure (caller falls back to full deserialization).
fn extract_request_id_fast(payload: &[u8]) -> Option<&str> {
    if payload.is_empty() {
        return None;
    }

    let marker = rmp::decode::read_marker(&mut &payload[..]).ok()?;

    match marker {
        // Array format follows WorkResult field order:
        // work_item_id, request_id, item_index, ...
        Marker::FixArray(n) if n >= 2 => {
            // Marker byte consumed 1 byte
            let data = skip_msgpack_value(&payload[1..])?;
            let (request_id, _) = read_str_from_slice(data).ok()?;
            Some(request_id)
        }
        Marker::Array16 => {
            // 1 marker byte + 2 length bytes = 3 bytes header
            if payload.len() < 3 {
                return None;
            }
            let len = u16::from_be_bytes([payload[1], payload[2]]);
            if len < 2 {
                return None;
            }
            let data = skip_msgpack_value(&payload[3..])?;
            let (request_id, _) = read_str_from_slice(data).ok()?;
            Some(request_id)
        }
        Marker::Array32 => {
            // 1 marker byte + 4 length bytes = 5 bytes header
            if payload.len() < 5 {
                return None;
            }
            let len = u32::from_be_bytes([payload[1], payload[2], payload[3], payload[4]]);
            if len < 2 {
                return None;
            }
            let data = skip_msgpack_value(&payload[5..])?;
            let (request_id, _) = read_str_from_slice(data).ok()?;
            Some(request_id)
        }

        // Map format: scan for "request_id" key
        Marker::FixMap(n) => scan_map_for_request_id(&payload[1..], n as u32),
        Marker::Map16 => {
            if payload.len() < 3 {
                return None;
            }
            let n = u16::from_be_bytes([payload[1], payload[2]]) as u32;
            scan_map_for_request_id(&payload[3..], n)
        }
        Marker::Map32 => {
            if payload.len() < 5 {
                return None;
            }
            let n = u32::from_be_bytes([payload[1], payload[2], payload[3], payload[4]]);
            scan_map_for_request_id(&payload[5..], n)
        }

        _ => None,
    }
}

/// Scan a msgpack map's key-value pairs for the "request_id" key.
fn scan_map_for_request_id(mut data: &[u8], num_entries: u32) -> Option<&str> {
    for _ in 0..num_entries {
        let (key, rest) = read_str_from_slice(data).ok()?;
        data = rest;

        if key == "request_id" {
            let (value, _) = read_str_from_slice(data).ok()?;
            return Some(value);
        }

        // Skip the value
        data = skip_msgpack_value(data)?;
    }
    None
}

/// Skip one msgpack value in the byte slice, returning the remaining bytes.
fn skip_msgpack_value(data: &[u8]) -> Option<&[u8]> {
    if data.is_empty() {
        return None;
    }

    let marker = rmp::decode::read_marker(&mut &data[..]).ok()?;
    let rest = &data[1..]; // after marker byte

    match marker {
        Marker::Null | Marker::True | Marker::False => Some(rest),
        Marker::FixPos(_) | Marker::FixNeg(_) => Some(rest),

        Marker::U8 | Marker::I8 => rest.get(1..),
        Marker::U16 | Marker::I16 => rest.get(2..),
        Marker::U32 | Marker::I32 | Marker::F32 => rest.get(4..),
        Marker::U64 | Marker::I64 | Marker::F64 => rest.get(8..),

        Marker::FixStr(len) => rest.get(len as usize..),
        Marker::Str8 => {
            let len = *rest.first()? as usize;
            rest.get(1usize.checked_add(len)?..)
        }
        Marker::Str16 => {
            if rest.len() < 2 {
                return None;
            }
            let len = u16::from_be_bytes([rest[0], rest[1]]) as usize;
            rest.get(2usize.checked_add(len)?..)
        }
        Marker::Str32 => {
            if rest.len() < 4 {
                return None;
            }
            let len = u32::from_be_bytes([rest[0], rest[1], rest[2], rest[3]]) as usize;
            rest.get(4usize.checked_add(len)?..)
        }

        Marker::Bin8 => {
            let len = *rest.first()? as usize;
            rest.get(1usize.checked_add(len)?..)
        }
        Marker::Bin16 => {
            if rest.len() < 2 {
                return None;
            }
            let len = u16::from_be_bytes([rest[0], rest[1]]) as usize;
            rest.get(2usize.checked_add(len)?..)
        }
        Marker::Bin32 => {
            if rest.len() < 4 {
                return None;
            }
            let len = u32::from_be_bytes([rest[0], rest[1], rest[2], rest[3]]) as usize;
            rest.get(4usize.checked_add(len)?..)
        }

        Marker::FixArray(n) => {
            let mut d = rest;
            for _ in 0..n {
                d = skip_msgpack_value(d)?;
            }
            Some(d)
        }
        Marker::Array16 => {
            if rest.len() < 2 {
                return None;
            }
            let n = u16::from_be_bytes([rest[0], rest[1]]) as u32;
            let mut d = &rest[2..];
            for _ in 0..n {
                d = skip_msgpack_value(d)?;
            }
            Some(d)
        }
        Marker::Array32 => {
            if rest.len() < 4 {
                return None;
            }
            let n = u32::from_be_bytes([rest[0], rest[1], rest[2], rest[3]]);
            let mut d = &rest[4..];
            for _ in 0..n {
                d = skip_msgpack_value(d)?;
            }
            Some(d)
        }

        Marker::FixMap(n) => {
            let mut d = rest;
            for _ in 0..n {
                d = skip_msgpack_value(d)?;
                d = skip_msgpack_value(d)?;
            }
            Some(d)
        }
        Marker::Map16 => {
            if rest.len() < 2 {
                return None;
            }
            let n = u16::from_be_bytes([rest[0], rest[1]]) as u32;
            let mut d = &rest[2..];
            for _ in 0..n {
                d = skip_msgpack_value(d)?;
                d = skip_msgpack_value(d)?;
            }
            Some(d)
        }
        Marker::Map32 => {
            if rest.len() < 4 {
                return None;
            }
            let n = u32::from_be_bytes([rest[0], rest[1], rest[2], rest[3]]);
            let mut d = &rest[4..];
            for _ in 0..n {
                d = skip_msgpack_value(d)?;
                d = skip_msgpack_value(d)?;
            }
            Some(d)
        }

        Marker::FixExt1 => rest.get(2..),
        Marker::FixExt2 => rest.get(3..),
        Marker::FixExt4 => rest.get(5..),
        Marker::FixExt8 => rest.get(9..),
        Marker::FixExt16 => rest.get(17..),
        Marker::Ext8 => {
            let len = *rest.first()? as usize;
            rest.get(2usize.checked_add(len)?..)
        }
        Marker::Ext16 => {
            if rest.len() < 2 {
                return None;
            }
            let len = u16::from_be_bytes([rest[0], rest[1]]) as usize;
            rest.get(3usize.checked_add(len)?..)
        }
        Marker::Ext32 => {
            if rest.len() < 4 {
                return None;
            }
            let len = u32::from_be_bytes([rest[0], rest[1], rest[2], rest[3]]) as usize;
            rest.get(5usize.checked_add(len)?..)
        }

        Marker::Reserved => None,
    }
}

/// Outcome of [`WorkPublisher::republish_to_pool_outcome`].
///
/// The legacy [`WorkPublisher::republish_to_pool`] wrapper collapses the
/// non-`Republished` arms to `false`, but `handle_nak` needs to tell
/// "already republished" apart from "no collector / no payload": a NAK
/// that arrives after the request already fell back to the pool means the
/// item has been retried as far as it can be, so the client should get a
/// 429 immediately rather than hang until the first-chunk timeout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RepublishOutcome {
    /// The item was re-issued to the pool subject.
    Republished,
    /// A prior NAK/timeout already republished this request; nothing was
    /// done now.
    AlreadyRepublished,
    /// Nothing could be republished: no live collector, no cached
    /// payload, or no fallback subject.
    NotPossible,
    /// H9 — the per-(model, pool) first-chunk-fallback token bucket
    /// was empty; the republish was deliberately refused so the
    /// fallback rate stays bounded under a cold-start storm. Callers
    /// should surface a 504 to the client (the request has already
    /// failed-over once; we won't safely re-fallback now).
    RateLimited,
}

impl WorkPublisher {
    pub fn new(
        jetstream: jetstream::Context,
        router_id: String,
        payload_store: Arc<dyn PayloadStore>,
        result_timeout: Duration,
        max_stream_pending: u64,
        stream_max_age: Duration,
    ) -> Self {
        Self {
            jetstream,
            router_id,
            payload_store,
            result_timeout,
            max_stream_pending,
            stream_max_age,
            ensured_streams: DashMap::new(),
            stream_info_cache: DashMap::new(),
            pending_results: DashMap::new(),
            pending_streams: DashMap::new(),
            offloaded_streams: DashSet::new(),
            offloaded_payload_keys: DashMap::new(),
            nats_client: tokio::sync::RwLock::new(None),
            inbox_handle: tokio::sync::Mutex::new(None),
            fallback_buckets: DashMap::new(),
            fallback_rate_per_sec: FALLBACK_RATE_PER_SEC_DEFAULT,
            fallback_burst: FALLBACK_BURST_DEFAULT,
        }
    }

    /// Try to consume one first-chunk-fallback token for ``(model, pool)``.
    /// Returns true when a republish is permitted, false when the bucket is
    /// empty (the caller MUST surface a 504 / refused-republish outcome).
    /// Helper exposed so the `first_chunk_timeout`-driven call sites in
    /// `republish_to_pool_outcome` and any future fallback entry-points
    /// share the same bucket state.
    fn try_take_fallback_token(&self, model: &str, pool: &str) -> bool {
        let key = format!("{}|{}", model, pool);
        let bucket = self.fallback_buckets.entry(key).or_insert_with(|| {
            std::sync::Mutex::new(TokenBucket::new(
                self.fallback_rate_per_sec,
                self.fallback_burst,
            ))
        });
        // Lock is a plain (non-async) ``std::sync::Mutex``; the critical
        // section is a few additions + a comparison, so we never hold it
        // across an ``await``. Poisoning means an earlier panic in the
        // critical section — treat as "deny" so a panicked process state
        // doesn't silently allow unbounded fallbacks.
        let mut guard = match bucket.value().lock() {
            Ok(g) => g,
            Err(_) => return false,
        };
        guard.try_take()
    }

    #[allow(dead_code)]
    pub fn router_id(&self) -> &str {
        &self.router_id
    }

    /// Ensure the stream exists for the given pool (cached — admin call happens once per pool).
    ///
    /// Returns the cached JetStream stream name as an `Arc<str>` so
    /// callers can avoid rebuilding `format!("WORK_POOL_{pool}")` on
    /// the hot path.
    pub async fn ensure_stream(&self, pool: &str) -> Result<Arc<str>, String> {
        if let Some(existing) = self.ensured_streams.get(pool) {
            return Ok(Arc::clone(&existing));
        }

        let name = stream_name(pool);
        let desired_subject = format!("sie.work.{}.*.*.*", pool);
        let subjects = vec![desired_subject.clone()];

        let mut stream = self
            .jetstream
            .get_or_create_stream(jetstream::stream::Config {
                name: name.clone(),
                subjects,
                retention: jetstream::stream::RetentionPolicy::WorkQueue,
                storage: jetstream::stream::StorageType::Memory,
                max_age: self.stream_max_age,
                max_messages: 100_000,
                ..Default::default()
            })
            .await
            .map_err(|e| format!("create/get stream {}: {}", name, e))?;
        let observed_subjects = stream.cached_info().config.subjects.clone();
        if let Some(updated_subjects) =
            canonical_stream_subjects(observed_subjects.clone(), &desired_subject)
        {
            let mut updated = stream.cached_info().config.clone();
            updated.subjects = updated_subjects;
            self.jetstream
                .update_stream(updated)
                .await
                .map_err(|e| format!("update stream {} subjects: {}", name, e))?;
            stream = self
                .jetstream
                .get_stream(name.clone())
                .await
                .map_err(|e| format!("refresh stream {} after subject update: {}", name, e))?;
            info!(
                stream = %name,
                observed_subjects = ?observed_subjects,
                desired_subject = %desired_subject,
                "reconciled JetStream stream subjects to canonical queue lane routing"
            );
        }
        let observed_max_age = stream.cached_info().config.max_age;
        if observed_max_age != self.stream_max_age {
            let mut updated = stream.cached_info().config.clone();
            updated.max_age = self.stream_max_age;
            self.jetstream
                .update_stream(updated)
                .await
                .map_err(|e| format!("update stream {} max_age: {}", name, e))?;
            stream = self
                .jetstream
                .get_stream(name.clone())
                .await
                .map_err(|e| format!("refresh stream {} after max_age update: {}", name, e))?;
            info!(
                stream = %name,
                observed_max_age_s = observed_max_age.as_secs(),
                desired_max_age_s = self.stream_max_age.as_secs(),
                "reconciled JetStream stream max_age"
            );
        }

        // Prime the backpressure cache so the very first request to
        // this pool sees real consumer/pending numbers instead of
        // fail-open-until-next-monitor-tick (up to ~tick ms window).
        // We swallow errors here: if the info call fails we fall
        // back to the old behaviour of allowing the first request
        // through, matching the pre-change semantics.
        match stream.info().await {
            Ok(info) => {
                self.stream_info_cache.insert(
                    pool.to_string(),
                    CachedStreamInfo {
                        num_pending: info.state.messages,
                        num_consumers: info.state.consumer_count,
                    },
                );
            }
            Err(e) => {
                debug!(
                    stream = %name,
                    error = %e,
                    "priming stream info cache failed; falling back to monitor tick"
                );
            }
        }

        let arc_name: Arc<str> = Arc::from(name.as_str());
        self.ensured_streams
            .insert(pool.to_string(), Arc::clone(&arc_name));
        info!(stream = %name, "ensured JetStream stream");
        Ok(arc_name)
    }

    /// Check backpressure from cached stream info (lock-free DashMap read).
    /// Actual NATS stream.info() calls happen in the background monitor task.
    ///
    /// **Direct-dispatch caveat:** this check measures the pool stream's
    /// pending count. When HRW routes to a worker's private subject
    /// (`sie.work.{pool}.{machine_profile}.{bundle}.{model}.{worker_id}`)
    /// the pool count can read low while the chosen worker's inbox is
    /// saturated. The first-chunk timeout in `proxy::stream_generate_response`
    /// is the safety net: on timeout, `republish_to_pool` redrives the work
    /// item onto the lane's pool subject so any healthy matching worker can
    /// pick it up. Tighter per-worker admission is M5+ work (tracked
    /// alongside the §6 mixed-pool fairness scheduler).
    fn check_backpressure(&self, pool: &str) -> Result<(), String> {
        if let Some(info) = self.stream_info_cache.get(pool) {
            if info.num_consumers == 0 {
                return Err("no consumers available for work stream".to_string());
            }
            if info.num_pending > self.max_stream_pending {
                return Err(format!(
                    "backpressure: {} pending messages exceeds threshold {}",
                    info.num_pending, self.max_stream_pending
                ));
            }
        }
        // No cached info yet (ensure_stream failed to prime) — allow
        // through; the monitor task will fill the cache shortly.
        Ok(())
    }

    /// Clear cached stream state on NATS reconnect.
    /// After a NATS server restart, streams may have been deleted. Without clearing,
    /// the gateway would publish to non-existent streams and requests would timeout.
    pub fn clear_caches(&self) {
        self.ensured_streams.clear();
        self.stream_info_cache.clear();
        info!("cleared ensured_streams and stream_info caches (NATS reconnect)");
    }

    /// Start background task that polls stream info for all known pools.
    ///
    /// The 50 ms tick (down from 200 ms) shortens the window during
    /// which `check_backpressure` sees a stale snapshot after a burst
    /// starts draining. At the QPS this gateway serves the extra
    /// `stream.info()` calls are negligible (one per pool per tick),
    /// and traded directly against tail-latency recovery time when
    /// consumers catch up. The first-hit window itself is now zero:
    /// `ensure_stream` primes the cache synchronously.
    pub fn start_backpressure_monitor(self: &Arc<Self>) {
        let publisher = Arc::clone(self);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(50));
            loop {
                interval.tick().await;

                // Snapshot (pool, stream_name) pairs so we don't have
                // to rebuild the stream name string for each pool on
                // every tick.
                let pools: Vec<(String, Arc<str>)> = publisher
                    .ensured_streams
                    .iter()
                    .map(|entry| (entry.key().clone(), Arc::clone(entry.value())))
                    .collect();

                for (pool, name) in pools {
                    let mut stream = match publisher.jetstream.get_stream(name.as_ref()).await {
                        Ok(s) => s,
                        Err(_) => continue,
                    };
                    let info = match stream.info().await {
                        Ok(i) => i,
                        Err(_) => continue,
                    };
                    publisher.stream_info_cache.insert(
                        pool,
                        CachedStreamInfo {
                            num_pending: info.state.messages,
                            num_consumers: info.state.consumer_count,
                        },
                    );
                }
            }
        });
    }

    /// Decompose a request into work items and publish to JetStream.
    #[allow(clippy::too_many_arguments)]
    pub async fn publish_work(
        &self,
        target: PublishTarget,
        admission_pool: &str,
        endpoint: &str,
        model: &str,
        engine: &str,
        bundle_config_hash: &str,
        items: Vec<rmpv::Value>,
        params: &WorkParams,
    ) -> Result<(String, oneshot::Receiver<Vec<WorkResult>>), String> {
        let start = Instant::now();
        let pool = target.pool().to_string();
        let gpu = target.machine_profile().to_string();

        // Ensure stream exists (cached — first call per pool does admin API, subsequent are free)
        self.ensure_stream(&pool).await?;

        // Check backpressure (lock-free read from background-updated cache)
        self.check_backpressure(&pool)?;

        // UUIDv7 keeps the leading 48 bits as a big-endian Unix
        // millisecond timestamp, so lexicographic / B-tree-indexed
        // storage of request_ids (JetStream subjects, DashMap keys,
        // downstream log aggregators) stays time-sortable without
        // extra fields. v4 gave us uniqueness but nothing else.
        let request_id = uuid::Uuid::now_v7().to_string();
        let reply_subject = format!("_INBOX.{}.{}", self.router_id, request_id);
        let total_items = if endpoint == "score" || endpoint == "generate" {
            1
        } else {
            items.len() as u32
        };

        let subject = target.subject();
        let pool_fallback_subject = match (&target, endpoint) {
            (PublishTarget::Worker { .. }, "generate") => None,
            (PublishTarget::Worker { .. }, _) => Some(target.as_pool_fallback().subject()),
            (PublishTarget::Pool { .. }, _) => None,
        };
        let direct_publish_fallback_subject = pool_fallback_subject.clone();
        let direct_publish_retry_items = if direct_publish_fallback_subject.is_some() {
            Some(items.clone())
        } else {
            None
        };

        // Set up result collector (DashMap — lock-free per-key insert)
        let (tx, rx) = oneshot::channel();
        self.pending_results.insert(
            request_id.clone(),
            ResultCollector {
                _total_items: total_items,
                results: vec![None; total_items as usize],
                sender: Some(tx),
                deadline: Instant::now() + self.result_timeout,
                operation: endpoint.to_string(),
                published_at: Instant::now(),
                pool_label: pool.clone(),
                model_label: model.to_string(),
                pool_fallback_subject,
                direct_fallback_payloads: vec![None; total_items as usize],
                direct_fallback_republished: false,
            },
        );

        // Build and publish all work items, collecting ack futures for parallel await.
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        // Generation queue fallback carries trace context. Encode /
        // score / extract intentionally keep the pre-generation queue
        // hot path and wire shape: no OTel lookup, no envelope fields.
        let (traceparent, tracestate) = Self::trace_context_for_endpoint(endpoint);

        // Every work item in a request shares the same pool / model /
        // params block. `WorkItemRef` borrows these values so we don't
        // pay N × `Option<Vec<String>> / Option<serde_json::Value>`
        // deep clones on the hot path.
        let shared = Arc::new(WorkItemShared {
            request_id: &request_id,
            endpoint,
            model,
            pool: &pool,
            admission_pool,
            gpu: &gpu,
            engine,
            bundle_config_hash,
            router_id: &self.router_id,
            reply_subject: &reply_subject,
            params,
            timestamp,
            traceparent: traceparent.as_deref(),
            tracestate: tracestate.as_deref(),
        });

        // Running publishes concurrently via `try_join_all` means
        // a late-failing item can race ahead of earlier successes —
        // so on any error we have to unwind the collector entry
        // and whatever payloads the successful siblings already
        // wrote to the offload store, otherwise both leak until
        // the result-timeout sweep kicks in.
        let publish_outcome = self
            .publish_batch_items(endpoint, Arc::clone(&shared), total_items, items, &subject)
            .await;
        let published_items = match publish_outcome {
            Ok(items) => items,
            Err(e) => {
                if let (Some(fallback_subject), Some(retry_items)) =
                    (direct_publish_fallback_subject, direct_publish_retry_items)
                {
                    warn!(
                        request_id = %request_id,
                        subject = %subject,
                        fallback_subject = %fallback_subject,
                        error = %e,
                        "worker-direct publish failed; retrying non-streaming work on pool subject"
                    );
                    match self
                        .publish_batch_items(
                            endpoint,
                            Arc::clone(&shared),
                            total_items,
                            retry_items,
                            &fallback_subject,
                        )
                        .await
                    {
                        Ok(items) => {
                            metrics::ROUTING_FALLBACK_TOTAL
                                .with_label_values(&[
                                    &metrics::sanitize_model_label(model),
                                    &metrics::sanitize_label(&pool),
                                    "batch_direct_publish_error",
                                ])
                                .inc();
                            items
                        }
                        Err(fallback_err) => {
                            self.pending_results.remove(&request_id);
                            self.cleanup_offloaded_payloads(&request_id).await;
                            return Err(format!(
                                "{}; pool fallback publish failed: {}",
                                e, fallback_err
                            ));
                        }
                    }
                } else {
                    self.pending_results.remove(&request_id);
                    self.cleanup_offloaded_payloads(&request_id).await;
                    return Err(e);
                }
            }
        };
        let mut ack_futures = Vec::with_capacity(published_items.len());
        let should_cache_direct_fallback = self
            .pending_results
            .get(&request_id)
            .is_some_and(|entry| entry.pool_fallback_subject.is_some());
        if should_cache_direct_fallback {
            if let Some(mut entry) = self.pending_results.get_mut(&request_id) {
                for item in &published_items {
                    if item.index < entry.direct_fallback_payloads.len() {
                        entry.direct_fallback_payloads[item.index] = Some(item.encoded.clone());
                    }
                }
            }
        }
        for item in published_items {
            ack_futures.push(item.ack);
        }

        // Fire-and-forget: spawn background task to monitor acks.
        // The request handler proceeds immediately to wait for inbox results.
        // JetStream acks confirm durability but our streams are ephemeral
        // (memory, 60s TTL) — clients retry on timeout if messages are lost.
        if !ack_futures.is_empty() {
            tokio::spawn(async move {
                for ack in ack_futures {
                    if let Err(e) = ack.await {
                        warn!(error = %e, "JetStream ack failed (message may be lost)");
                        metrics::QUEUE_ACK_FAILURES.inc();
                    }
                }
            });
        }

        let elapsed = start.elapsed();
        metrics::QUEUE_PUBLISH_SECONDS
            .with_label_values(&[endpoint])
            .observe(elapsed.as_secs_f64());
        metrics::QUEUE_ITEMS_PUBLISHED
            .with_label_values(&[endpoint])
            .observe(total_items as f64);

        debug!(
            request_id = %request_id,
            items = total_items,
            pool = %pool,
            target = target.label(),
            endpoint = %endpoint,
            latency_ms = elapsed.as_millis(),
            "published work items"
        );

        Ok((request_id, rx))
    }

    async fn publish_batch_items(
        &self,
        endpoint: &str,
        shared: Arc<WorkItemShared<'_>>,
        total_items: u32,
        items: Vec<rmpv::Value>,
        subject: &str,
    ) -> Result<Vec<PublishedWorkItem>, String> {
        if endpoint == "score" {
            self.publish_score(shared.as_ref(), items, subject)
                .await
                .map(|(ack, encoded)| {
                    vec![PublishedWorkItem {
                        index: 0,
                        ack,
                        encoded,
                    }]
                })
        } else if endpoint == "generate" {
            self.publish_generate(shared.as_ref(), subject)
                .await
                .map(|(ack, encoded)| {
                    vec![PublishedWorkItem {
                        index: 0,
                        ack,
                        encoded,
                    }]
                })
        } else {
            // JetStream publishes issue the send-and-receive-ack cycle
            // asynchronously per message, so a batch of N items that
            // used to serialize on `.await` now overlap their network
            // round trips. Each future in the set borrows `shared` +
            // owns its per-item `rmpv::Value`.
            let publishes = items.into_iter().enumerate().map(|(index, item_value)| {
                let subject = subject.to_string();
                let shared = Arc::clone(&shared);
                async move {
                    self.publish_single(shared.as_ref(), total_items, index, item_value, &subject)
                        .await
                        .map(|(ack, encoded)| PublishedWorkItem {
                            index,
                            ack,
                            encoded,
                        })
                }
            });
            try_join_all(publishes).await
        }
    }

    /// Arm a one-shot recovery for non-streaming worker-direct batch work.
    ///
    /// Capped logical pools direct-dispatch encode/score/extract to one
    /// admitted worker so unassigned peers do not burn the JetStream
    /// delivery budget. If that worker or its private consumer disappears
    /// after the gateway publishes, the worker-specific stream cannot fail
    /// over by itself. This timer republishes any still-missing items to the
    /// lane's pool subject once; late duplicate results are harmless because
    /// the result collector removes itself on first complete response.
    pub fn spawn_batch_direct_fallback(self: &Arc<Self>, request_id: String, delay: Duration) {
        let publisher = Arc::clone(self);
        tokio::spawn(async move {
            tokio::time::sleep(delay).await;
            if let Err(e) = publisher
                .republish_pending_result_to_pool(&request_id, "batch_direct_timeout")
                .await
            {
                warn!(
                    request_id = %request_id,
                    error = %e,
                    "batch direct-dispatch fallback failed"
                );
            }
        });
    }

    pub async fn republish_pending_result_to_pool(
        &self,
        request_id: &str,
        reason: &'static str,
    ) -> Result<bool, String> {
        let (subject, payloads, operation, pool, model) = {
            let Some(mut entry) = self.pending_results.get_mut(request_id) else {
                return Ok(false);
            };
            let Some((subject, payloads)) = take_direct_fallback_payloads(entry.value_mut()) else {
                return Ok(false);
            };
            let operation = entry.operation.clone();
            let pool = entry.pool_label.clone();
            let model = entry.model_label.clone();
            (subject, payloads, operation, pool, model)
        };

        let mut acks = Vec::with_capacity(payloads.len());
        for (index, payload) in payloads {
            let ack = self
                .jetstream
                .publish(subject.clone(), payload.into())
                .await
                .map_err(|e| {
                    format!(
                        "publish batch fallback item {} for {}: {}",
                        index, request_id, e
                    )
                })?;
            acks.push(ack);
        }

        metrics::ROUTING_FALLBACK_TOTAL
            .with_label_values(&[
                &metrics::sanitize_model_label(&model),
                &metrics::sanitize_label(&pool),
                reason,
            ])
            .inc();
        info!(
            request_id = %request_id,
            operation = %operation,
            subject = %subject,
            reason,
            "republished non-streaming direct-dispatch work to pool fallback"
        );

        for ack in acks {
            if let Err(e) = ack.await {
                warn!(
                    request_id = %request_id,
                    error = %e,
                    "JetStream ack failed for batch direct fallback"
                );
                metrics::QUEUE_ACK_FAILURES.inc();
            }
        }

        Ok(true)
    }

    fn trace_context_for_endpoint(endpoint: &str) -> (Option<String>, Option<String>) {
        if Self::should_propagate_queue_trace(endpoint) {
            crate::observability::propagation::inject_current_context()
        } else {
            (None, None)
        }
    }

    fn should_propagate_queue_trace(endpoint: &str) -> bool {
        InferenceEndpoint::from_label(endpoint).injects_queue_trace_context()
    }

    /// Publish the single work item for a score request.
    ///
    /// The score endpoint collapses the whole request into one work
    /// item that carries the query + all candidate items, so there's
    /// no per-item fan-out here — but we still route it through the
    /// shared borrow helper to keep a single code path for encoding.
    async fn publish_score(
        &self,
        shared: &WorkItemShared<'_>,
        score_items: Vec<rmpv::Value>,
        subject: &str,
    ) -> Result<(jetstream::context::PublishAckFuture, Vec<u8>), String> {
        let query_item = shared
            .params
            .query_item
            .as_ref()
            .ok_or_else(|| "score request missing query item".to_string())?;

        let work_item_id = format!("{}.0", shared.request_id);
        let ref_item = WorkItemRef {
            work_item_id: &work_item_id,
            request_id: shared.request_id,
            item_index: 0,
            total_items: 1,
            operation: shared.endpoint,
            model_id: shared.model,
            profile_id: "default",
            engine: shared.engine,
            pool_name: shared.pool,
            admission_pool: shared.admission_pool,
            machine_profile: shared.gpu,
            item: None,
            payload_ref: None,
            output_types: shared.params.output_types.as_deref(),
            instruction: shared.params.instruction.as_deref(),
            is_query: shared.params.is_query,
            options: shared.params.options.as_ref(),
            query_item: Some(query_item),
            query_payload_ref: None,
            score_items: Some(&score_items),
            labels: shared.params.labels.as_deref(),
            output_schema: shared.params.output_schema.as_ref(),
            generate: shared.params.generate.as_ref(),
            routing_key: shared.params.routing_key.as_deref(),
            prompt_cache_key: shared.params.prompt_cache_key.as_deref(),
            bundle_config_hash: shared.bundle_config_hash,
            router_id: shared.router_id,
            reply_subject: shared.reply_subject,
            timestamp: shared.timestamp,
            traceparent: shared.traceparent,
            tracestate: shared.tracestate,
        };

        let mut encoded =
            rmp_serde::to_vec_named(&ref_item).map_err(|e| format!("msgpack encode: {}", e))?;

        if encoded.len() > PAYLOAD_OFFLOAD_THRESHOLD {
            // Build the offloaded `{query, items}` envelope by
            // borrowing the already-decoded values (no deep clone of
            // the score_items array). We only pay one extra msgpack
            // encode — the same one we'd pay before — and the resulting
            // `WorkItem` on the wire is far smaller because the items
            // live in object storage.
            let score_payload_value = rmpv::Value::Map(vec![
                (rmpv::Value::from("query"), query_item.clone()),
                (
                    rmpv::Value::from("items"),
                    rmpv::Value::Array(score_items.clone()),
                ),
            ]);
            let score_payload = rmp_serde::to_vec_named(&score_payload_value)
                .map_err(|e| format!("msgpack encode score payload: {}", e))?;
            let ref_key = format!("{}_score.bin", shared.request_id);
            // Track before the store write (issue #1471 review): a future dropped
            // mid-`put` may still have landed the blob, so it must stay eligible
            // for cleanup.
            record_offloaded_payload_key(&self.offloaded_payload_keys, shared.request_id, &ref_key);
            if let Err(e) = self.payload_store.put(&ref_key, &score_payload).await {
                warn!(error = %e, "failed to offload score payload, sending inline");
            } else {
                let offloaded = WorkItemRef {
                    query_item: None,
                    query_payload_ref: Some(&ref_key),
                    score_items: None,
                    ..ref_item
                };
                encoded = rmp_serde::to_vec_named(&offloaded)
                    .map_err(|e| format!("msgpack encode offloaded score: {}", e))?;
                metrics::QUEUE_PAYLOAD_OFFLOADS.inc();
            }
        }

        let ack = self
            .jetstream
            .publish(subject.to_string(), encoded.clone().into())
            .await
            .map_err(|e| format!("publish score work item: {}", e))?;
        Ok((ack, encoded))
    }

    /// Publish a generation work item and wire up a streaming collector.
    /// Replaces the walking-skeleton path: instead of a one-shot
    /// ``WorkResult`` collector, this installs a [`StreamCollector`]
    /// that accumulates per-chunk envelopes and fires a
    /// [`StreamOutcome`] when the worker emits a terminal chunk.
    ///
    /// Returns the request id plus a receiver for the outcome. The
    /// caller should await with whatever timeout / cancellation logic
    /// is appropriate; the gateway-side timeout taxonomy lives in
    /// ``handlers/proxy.rs`` (Phase F).
    #[allow(clippy::too_many_arguments)]
    /// ``display_model`` is the requested base id surfaced on all
    /// metric labels for this request; ``target.model()`` is the
    /// dispatch id (a ``:no-spec`` grammar variant when routing fired)
    /// used for the NATS subject / work item. Equal unless routing fired.
    pub async fn publish_generate_streaming(
        &self,
        target: PublishTarget,
        display_model: &str,
        engine: &str,
        bundle_config_hash: &str,
        params: &WorkParams,
        admission_pool: &str,
    ) -> Result<
        (
            String,
            oneshot::Receiver<StreamOutcome>,
            std::sync::Arc<tokio::sync::Notify>,
        ),
        String,
    > {
        let model = target.model().to_string();
        let pool = target.pool().to_string();
        let machine_profile = target.machine_profile().to_string();
        // Reuse the same JetStream stream + backpressure plumbing as
        // the batch path. The stream collector replaces ResultCollector.
        self.ensure_stream(&pool).await?;
        self.check_backpressure(&pool)?;

        if params.generate.is_none() {
            return Err("generate request missing 'prompt' / 'max_new_tokens'".to_string());
        }

        let request_id = uuid::Uuid::now_v7().to_string();
        let reply_subject = format!("_INBOX.{}.{}", self.router_id, request_id);
        let subject = target.subject();
        let pool_fallback_subject = target.as_pool_fallback().subject();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        // Set up the streaming aggregator before publishing the work so
        // that a fast worker can't beat us to the inbox.
        let (tx, rx) = oneshot::channel::<StreamOutcome>();
        let mut collector = StreamCollector::new(tx, model.clone(), pool.clone());
        // Metric labels surface the requested (display) id, never the
        // ``:no-spec`` dispatch variant (#1324).
        collector.display_model = display_model.to_string();
        collector.pool_fallback_subject = Some(pool_fallback_subject.clone());
        // Capture the activity handle before the collector moves into
        // ``pending_streams`` so the caller never has to re-look it up
        // (eliminates the .expect() race window where an error path
        // could remove the collector before the handler retrieves it).
        let activity = collector.activity_handle();
        self.pending_streams.insert(request_id.clone(), collector);

        // M5: capture the W3C Trace Context once for envelope injection.
        let (traceparent, tracestate) = crate::observability::propagation::inject_current_context();

        let shared = WorkItemShared {
            request_id: &request_id,
            endpoint: "generate",
            model: &model,
            pool: &pool,
            admission_pool,
            gpu: &machine_profile,
            engine,
            bundle_config_hash,
            router_id: &self.router_id,
            reply_subject: &reply_subject,
            params,
            timestamp,
            traceparent: traceparent.as_deref(),
            tracestate: tracestate.as_deref(),
        };

        let publish_started = Instant::now();
        let (ack, encoded) = match self.publish_generate(&shared, &subject).await {
            Ok(pair) => pair,
            Err(e) => {
                self.pending_streams.remove(&request_id);
                return Err(e);
            }
        };
        // Cache the encoded payload on the collector so
        // ``republish_to_pool`` can re-issue the same item without
        // re-running the full serialization pipeline.
        if let Some(mut entry) = self.pending_streams.get_mut(&request_id) {
            entry.encoded_payload = Some(encoded);
        }

        // Fire-and-forget ack monitoring, matching the encode/score path.
        tokio::spawn(async move {
            if let Err(e) = ack.await {
                warn!(error = %e, "JetStream ack failed for generate (message may be lost)");
                metrics::QUEUE_ACK_FAILURES.inc();
            }
        });

        metrics::QUEUE_PUBLISH_SECONDS
            .with_label_values(&["generate"])
            .observe(publish_started.elapsed().as_secs_f64());
        metrics::QUEUE_ITEMS_PUBLISHED
            .with_label_values(&["generate"])
            .observe(1.0);

        Ok((request_id, rx, activity))
    }

    /// Streaming-SSE variant of [`Self::publish_generate_streaming`].
    /// Installs a per-chunk broadcast tap on the collector *before*
    /// publishing the work item, so an SSE handler can subscribe to
    /// every non-stale chunk delivered through ``handle_chunk``
    /// without racing against early arrivals. Returns the request id,
    /// the terminal-outcome receiver (unchanged from the
    /// non-SSE path — the SSE handler uses the broadcast receiver for
    /// per-chunk forwarding, but the oneshot remains the canonical
    /// completion signal for cancel-guard defuse and timing
    /// accounting), and the broadcast receiver to forward chunks.
    #[allow(clippy::too_many_arguments)]
    pub async fn publish_generate_streaming_sse(
        &self,
        target: PublishTarget,
        display_model: &str,
        engine: &str,
        bundle_config_hash: &str,
        params: &WorkParams,
        admission_pool: &str,
    ) -> Result<
        (
            String,
            oneshot::Receiver<StreamOutcome>,
            broadcast::Receiver<ChunkEnvelope>,
        ),
        String,
    > {
        let model = target.model().to_string();
        let pool = target.pool().to_string();
        let machine_profile = target.machine_profile().to_string();
        self.ensure_stream(&pool).await?;
        self.check_backpressure(&pool)?;

        if params.generate.is_none() {
            return Err("generate request missing 'prompt' / 'max_new_tokens'".to_string());
        }

        let request_id = uuid::Uuid::now_v7().to_string();
        let reply_subject = format!("_INBOX.{}.{}", self.router_id, request_id);
        let subject = target.subject();
        let pool_fallback_subject = target.as_pool_fallback().subject();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        // Install the collector + broadcast tap atomically before
        // publishing so no chunk envelope can race past the
        // subscriber.
        let (tx, rx) = oneshot::channel::<StreamOutcome>();
        let mut collector = StreamCollector::new(tx, model.clone(), pool.clone());
        // Metric labels surface the requested (display) id, never the
        // ``:no-spec`` dispatch variant (#1324).
        collector.display_model = display_model.to_string();
        collector.pool_fallback_subject = Some(pool_fallback_subject.clone());
        let chunk_rx = collector.install_chunk_tap();
        self.pending_streams.insert(request_id.clone(), collector);

        // M5: capture the W3C Trace Context once for envelope injection.
        let (traceparent, tracestate) = crate::observability::propagation::inject_current_context();

        let shared = WorkItemShared {
            request_id: &request_id,
            endpoint: "generate",
            model: &model,
            pool: &pool,
            admission_pool,
            gpu: &machine_profile,
            engine,
            bundle_config_hash,
            router_id: &self.router_id,
            reply_subject: &reply_subject,
            params,
            timestamp,
            traceparent: traceparent.as_deref(),
            tracestate: tracestate.as_deref(),
        };

        let publish_started = Instant::now();
        let (ack, encoded) = match self.publish_generate(&shared, &subject).await {
            Ok(pair) => pair,
            Err(e) => {
                self.pending_streams.remove(&request_id);
                return Err(e);
            }
        };
        if let Some(mut entry) = self.pending_streams.get_mut(&request_id) {
            entry.encoded_payload = Some(encoded);
        }
        tokio::spawn(async move {
            if let Err(e) = ack.await {
                warn!(error = %e, "JetStream ack failed for generate-sse (message may be lost)");
                metrics::QUEUE_ACK_FAILURES.inc();
            }
        });

        metrics::QUEUE_PUBLISH_SECONDS
            .with_label_values(&["generate"])
            .observe(publish_started.elapsed().as_secs_f64());
        metrics::QUEUE_ITEMS_PUBLISHED
            .with_label_values(&["generate"])
            .observe(1.0);

        Ok((request_id, rx, chunk_rx))
    }

    /// Forcibly drop a streaming collector — used by the HTTP handler
    /// when the client disconnects or a timeout fires, so the inbox
    /// subscriber stops accumulating chunks that nobody will read.
    pub fn drop_pending_stream(&self, request_id: &str) {
        self.pending_streams.remove(request_id);
    }

    pub fn pending_generation_snapshot(&self) -> PendingGenerationSnapshot {
        let now = Instant::now();
        let mut grouped: BTreeMap<(String, String, String), PendingGenerationGroup> =
            BTreeMap::new();

        for entry in self.pending_streams.iter() {
            accumulate_pending_generation_group(&mut grouped, entry.value(), now);
        }

        let groups: Vec<PendingGenerationGroup> = grouped.into_values().collect();
        PendingGenerationSnapshot {
            total: groups.iter().map(|group| group.count).sum(),
            groups,
        }
    }

    pub fn pending_generation_for_model(&self, model_id: &str) -> PendingGenerationSnapshot {
        self.pending_generation_snapshot().for_model(model_id)
    }

    /// Terminate an in-flight streaming request with a synthetic
    /// error outcome and fire the result sender so the HTTP handler
    /// returns immediately. Used by :meth:`handle_nak` when both the
    /// direct-dispatched worker NAKed *and* the pool republish failed
    /// — the request is unrecoverable, and the existing first-chunk
    /// timeout would otherwise make the client wait the full window
    /// for a response we already know will fail.
    ///
    /// The synthesised :class:`StreamOutcome` carries ``error =
    /// Some({code, message})``; the HTTP handler maps the code to
    /// the canonical OpenAI error envelope and HTTP status.
    fn fail_pending_stream(&self, request_id: &str, code: &str, message: &str) {
        let Some((_, mut collector)) = self.pending_streams.remove(request_id) else {
            return;
        };
        let outcome = crate::queue::streaming::StreamOutcome {
            text: String::new(),
            finish_reason: "error".to_string(),
            usage: None,
            attempt_id: collector.current_attempt_id.clone().unwrap_or_default(),
            ttft_ms: None,
            tpot_ms: None,
            error: Some(crate::queue::streaming::ChunkError {
                code: code.to_string(),
                message: message.to_string(),
            }),
            tool_calls: None,
            logprobs: None,
            candidates: Vec::new(),
        };
        if let Some(sender) = collector.sender.take() {
            if sender.send(outcome).is_err() {
                debug!(
                    request_id = %request_id,
                    "terminal-error outcome receiver dropped (client likely disconnected)"
                );
            }
        }
    }

    /// Cancel signal. Publishes an empty message on the core
    /// NATS subject ``cancel.{router_id}.{request_id}`` so any worker
    /// currently processing this request stops driving the adapter and
    /// emits a terminal ``finish_reason: "cancelled"`` chunk. Best-
    /// effort: silently no-ops if the core NATS client is not yet
    /// available (deployments without an active NATS connection).
    pub async fn publish_cancel(&self, request_id: &str) {
        let client_opt = { self.nats_client.read().await.clone() };
        let Some(client) = client_opt else {
            return;
        };
        let subject = format!("cancel.{}.{}", self.router_id, request_id);
        if let Err(e) = client.publish(subject, Vec::new().into()).await {
            warn!(error = %e, request_id = %request_id, "failed to publish cancel signal");
        }
    }

    /// Returns ``true`` iff the streaming collector for ``request_id``
    /// has already observed at least one chunk — used to label
    /// cancellation metrics (`before_first_chunk` vs `mid_stream`).
    pub fn stream_observed_first_chunk(&self, request_id: &str) -> bool {
        self.pending_streams
            .get(request_id)
            .map(|entry| entry.value().first_chunk_at.is_some())
            .unwrap_or(false)
    }

    /// Snapshot of the collector's timing state for inter-chunk timeout
    /// arming. Returns ``(first_chunk_at, last_chunk_at)``.
    pub fn stream_chunk_timing(
        &self,
        request_id: &str,
    ) -> Option<(Option<Instant>, Option<Instant>)> {
        self.pending_streams
            .get(request_id)
            .map(|entry| (entry.value().first_chunk_at, entry.value().last_chunk_at))
    }

    /// Publish a single generation work item (walking-skeleton path).
    ///
    /// Mirrors ``publish_score``: one WorkItem per HTTP request, no fan-out.
    /// All sampling fields live under ``shared.params.generate`` and travel
    /// through the typed ``WorkItemRef.generate`` field; the worker reads
    /// them via ``WorkItem.get("generate")`` in
    /// ``processors/streaming.py:StreamingProcessor.process``.
    async fn publish_generate(
        &self,
        shared: &WorkItemShared<'_>,
        subject: &str,
    ) -> Result<(jetstream::context::PublishAckFuture, Vec<u8>), String> {
        if shared.params.generate.is_none() {
            return Err("generate request missing 'prompt' / 'max_new_tokens'".to_string());
        }

        let work_item_id = format!("{}.0", shared.request_id);
        let mut ref_item = WorkItemRef {
            work_item_id: &work_item_id,
            request_id: shared.request_id,
            item_index: 0,
            total_items: 1,
            operation: shared.endpoint,
            model_id: shared.model,
            profile_id: "default",
            engine: shared.engine,
            pool_name: shared.pool,
            admission_pool: shared.admission_pool,
            machine_profile: shared.gpu,
            item: None,
            payload_ref: None,
            output_types: shared.params.output_types.as_deref(),
            instruction: shared.params.instruction.as_deref(),
            is_query: shared.params.is_query,
            options: shared.params.options.as_ref(),
            query_item: None,
            query_payload_ref: None,
            score_items: None,
            labels: shared.params.labels.as_deref(),
            output_schema: shared.params.output_schema.as_ref(),
            generate: shared.params.generate.as_ref(),
            routing_key: shared.params.routing_key.as_deref(),
            prompt_cache_key: shared.params.prompt_cache_key.as_deref(),
            bundle_config_hash: shared.bundle_config_hash,
            router_id: shared.router_id,
            reply_subject: shared.reply_subject,
            timestamp: shared.timestamp,
            traceparent: shared.traceparent,
            tracestate: shared.tracestate,
        };

        let mut encoded =
            rmp_serde::to_vec_named(&ref_item).map_err(|e| format!("msgpack encode: {}", e))?;

        // Offload large generate work items (e.g. multi-MB document images) to
        // the object store so they don't exceed NATS's max payload — mirrors
        // the encode/score paths. The blob is the ``generate`` params alone
        // (base64-string image data, so it stays serde_json::Value-decodable on
        // the sidecar). ``payload_ref`` lives at ``{request_id}_0.bin``; the
        // sidecar fetches + inlines it before handing the work item to Python.
        let offload_key;
        if encoded.len() > PAYLOAD_OFFLOAD_THRESHOLD {
            let generate = shared
                .params
                .generate
                .as_ref()
                .ok_or_else(|| "generate request missing params for offload".to_string())?;
            let gen_blob = rmp_serde::to_vec_named(generate)
                .map_err(|e| format!("msgpack encode offloaded generate: {}", e))?;
            offload_key = format!("{}_0.bin", shared.request_id);
            if let Err(e) = self.payload_store.put(&offload_key, &gen_blob).await {
                warn!(error = %e, "failed to offload generate payload, sending inline");
            } else {
                ref_item.generate = None;
                ref_item.payload_ref = Some(&offload_key);
                encoded = rmp_serde::to_vec_named(&ref_item)
                    .map_err(|e| format!("msgpack encode offloaded generate item: {}", e))?;
                self.offloaded_streams.insert(shared.request_id.to_string());
                metrics::QUEUE_PAYLOAD_OFFLOADS.inc();
            }
        }

        // Defense-in-depth dedup: stamp `Nats-Msg-Id` = request_id so a
        // gateway-side retry (or a duplicate publish racing on the same
        // request) collapses to a single message inside the stream's
        // dedup window. This is the *initial* attempt, so the bare
        // request_id is the right identity; the pool-republish path uses
        // a generation-suffixed id (see `republish_to_pool`) precisely so
        // it is NOT swallowed as a duplicate of this publish.
        let ack = self
            .jetstream
            .send_publish(
                subject.to_string(),
                jetstream::message::PublishMessage::build()
                    .message_id(shared.request_id)
                    .payload(encoded.clone().into()),
            )
            .await
            .map_err(|e| format!("publish generate work item: {}", e))?;
        Ok((ack, encoded))
    }

    /// Re-issue a previously-published generation item to the
    /// pool subject after a NAK or first-chunk timeout. Idempotent via
    /// the per-collector ``republished`` guard so a concurrent NAK +
    /// timeout race cannot double-publish.
    ///
    /// Returns `Ok(true)` if the item was republished, `Ok(false)` if
    /// nothing was done (no collector, no cached payload, or already
    /// republished), and `Err` only on a NATS publish failure.
    ///
    /// Thin wrapper over [`Self::republish_to_pool_outcome`] that keeps
    /// the historical `bool` contract for callers (the first-chunk-timeout
    /// paths in `proxy.rs` / `sse.rs`) that only need "did we republish".
    ///
    /// H9 — a refused republish (token bucket empty) collapses to
    /// `Ok(false)` for backward compatibility; callers that need to
    /// distinguish "refused / rate-limited" from "nothing to do"
    /// should call [`Self::republish_to_pool_status`] instead. The
    /// metric `sie_gateway_generation_fallback_refused_total` records
    /// the rate-limit case independently of either contract.
    pub async fn republish_to_pool(
        &self,
        request_id: &str,
        reason: &'static str,
    ) -> Result<bool, String> {
        Ok(matches!(
            self.republish_to_pool_outcome(request_id, reason).await?,
            RepublishOutcome::Republished
        ))
    }

    /// Three-state variant of [`Self::republish_to_pool`] for the
    /// first-chunk-timeout call sites that need to distinguish "rate
    /// limited" (surface a 504) from "nothing was republished" (surface
    /// the underlying first_chunk timeout). Returns ``Ok(true)`` on a
    /// successful republish, ``Ok(false)`` on any non-rate-limit
    /// no-op, and ``Err(...)`` for the rate-limit refusal so the call
    /// site doesn't have to introduce a new sentinel type. The error
    /// string is wire-stable: ``"fallback_rate_limited"``.
    #[allow(dead_code)] // reserved for future proxy.rs/sse.rs adoption (H9)
    pub async fn republish_to_pool_or_rate_limited(
        &self,
        request_id: &str,
        reason: &'static str,
    ) -> Result<bool, String> {
        match self.republish_to_pool_outcome(request_id, reason).await? {
            RepublishOutcome::Republished => Ok(true),
            RepublishOutcome::RateLimited => Err("fallback_rate_limited".to_string()),
            RepublishOutcome::AlreadyRepublished | RepublishOutcome::NotPossible => Ok(false),
        }
    }

    /// Republish variant that distinguishes "already republished" from
    /// "nothing to republish" (see [`RepublishOutcome`]). `handle_nak`
    /// uses it to surface a 429 when a NAK lands on a request that has
    /// already fallen back to the pool.
    async fn republish_to_pool_outcome(
        &self,
        request_id: &str,
        reason: &'static str,
    ) -> Result<RepublishOutcome, String> {
        // Take a short lock to: check + flip `republished`, copy out
        // the encoded bytes + the pool subject, bump the attempt
        // generation. We drop the entry lock before awaiting the
        // JetStream publish to avoid holding the DashMap shard.
        let (subject, payload, generation, display_model, pool) = {
            let Some(mut entry) = self.pending_streams.get_mut(request_id) else {
                return Ok(RepublishOutcome::NotPossible);
            };
            if entry.republished {
                return Ok(RepublishOutcome::AlreadyRepublished);
            }
            let Some(subject) = entry.pool_fallback_subject.clone() else {
                return Ok(RepublishOutcome::NotPossible);
            };
            let Some(payload) = entry.encoded_payload.clone() else {
                return Ok(RepublishOutcome::NotPossible);
            };
            // H9 — gate the first-chunk-fallback path on a per-(model,
            // pool) token bucket. Skipped for the NAK-driven reasons
            // because that path is already self-throttled by the
            // worker's own NAK rate; rate-limiting it again would
            // surface a noisy 429/504 during a single misbehaving
            // worker draining its queue. The check happens BEFORE we
            // flip ``entry.republished`` so a refused attempt can be
            // retried after the bucket refills (the deadline-armed
            // caller in proxy.rs/sse.rs already breaks out on the
            // refused signal — no spin).
            if reason == "first_chunk_timeout"
                && !self.try_take_fallback_token(entry.model.as_str(), entry.pool.as_str())
            {
                metrics::GENERATION_FALLBACK_REFUSED_TOTAL
                    .with_label_values(&[
                        &metrics::sanitize_model_label(entry.display_model.as_str()),
                        &metrics::sanitize_label(entry.pool.as_str()),
                        "rate_limited",
                    ])
                    .inc();
                return Ok(RepublishOutcome::RateLimited);
            }
            entry.republished = true;
            let gen = entry.bump_attempt_generation();
            (
                subject,
                payload,
                gen,
                entry.display_model.clone(),
                entry.pool.clone(),
            )
        };

        info!(
            request_id = %request_id,
            reason = reason,
            generation = generation,
            subject = %subject,
            "republishing generate item to pool"
        );

        // Defense-in-depth dedup: stamp a generation-suffixed
        // `Nats-Msg-Id`. It must differ from the initial publish's
        // `request_id` (and from any earlier republish) so JetStream's
        // dedup window does NOT swallow the republish as a duplicate of
        // the original — that would silently break the fallback. The
        // `attempt_generation` monotonically increases per republish, so
        // `{request_id}#{generation}` is unique per attempt while still
        // collapsing an accidental duplicate republish of the *same*
        // generation.
        let msg_id = format!("{}#{}", request_id, generation);
        match self
            .jetstream
            .send_publish(
                subject.clone(),
                jetstream::message::PublishMessage::build()
                    .message_id(&msg_id)
                    .payload(payload.into()),
            )
            .await
        {
            Ok(ack) => {
                metrics::ROUTING_FALLBACK_TOTAL
                    .with_label_values(&[
                        &metrics::sanitize_model_label(display_model.as_str()),
                        &metrics::sanitize_label(pool.as_str()),
                        reason,
                    ])
                    .inc();
                // Drive the ack to completion in the background; failure
                // here just becomes a delivery loss surfaced by the
                // existing inter-chunk timeout.
                tokio::spawn(async move {
                    if let Err(e) = ack.await {
                        warn!(error = %e, "JetStream ack failed for republish");
                        metrics::QUEUE_ACK_FAILURES.inc();
                    }
                });
                Ok(RepublishOutcome::Republished)
            }
            Err(e) => {
                // Roll back the `republished`/`attempt_generation`
                // flip so a downstream NAK or first-chunk-timeout can
                // retry. Without this, the request hangs until
                // overall-timeout: subsequent `republish_to_pool` calls
                // hit the `entry.republished` short-circuit at the top
                // and return `AlreadyRepublished`, but the worker never
                // received anything to begin with.
                if let Some(mut entry) = self.pending_streams.get_mut(request_id) {
                    entry.republished = false;
                    entry.rewind_attempt_generation();
                }
                Err(format!("republish to pool failed: {}", e))
            }
        }
    }

    /// Publish one work item of an encode / extract / other fan-out
    /// endpoint. Returns the per-item JetStream `PublishAckFuture` so
    /// callers can await acks in the background.
    async fn publish_single(
        &self,
        shared: &WorkItemShared<'_>,
        total_items: u32,
        index: usize,
        item_value: rmpv::Value,
        subject: &str,
    ) -> Result<(jetstream::context::PublishAckFuture, Vec<u8>), String> {
        let work_item_id = format!("{}.{}", shared.request_id, index);
        let mut ref_item = WorkItemRef {
            work_item_id: &work_item_id,
            request_id: shared.request_id,
            item_index: index as u32,
            total_items,
            operation: shared.endpoint,
            model_id: shared.model,
            profile_id: "default",
            engine: shared.engine,
            pool_name: shared.pool,
            admission_pool: shared.admission_pool,
            machine_profile: shared.gpu,
            item: Some(&item_value),
            payload_ref: None,
            output_types: shared.params.output_types.as_deref(),
            instruction: shared.params.instruction.as_deref(),
            is_query: shared.params.is_query,
            options: shared.params.options.as_ref(),
            query_item: None,
            query_payload_ref: None,
            score_items: None,
            labels: shared.params.labels.as_deref(),
            output_schema: shared.params.output_schema.as_ref(),
            generate: shared.params.generate.as_ref(),
            routing_key: shared.params.routing_key.as_deref(),
            prompt_cache_key: shared.params.prompt_cache_key.as_deref(),
            bundle_config_hash: shared.bundle_config_hash,
            router_id: shared.router_id,
            reply_subject: shared.reply_subject,
            timestamp: shared.timestamp,
            traceparent: shared.traceparent,
            tracestate: shared.tracestate,
        };

        let mut encoded =
            rmp_serde::to_vec_named(&ref_item).map_err(|e| format!("msgpack encode: {}", e))?;

        let offload_key;
        if encoded.len() > PAYLOAD_OFFLOAD_THRESHOLD {
            // Fail fast on encode error instead of `unwrap_or_default()`:
            // a silent empty blob in the payload store would only
            // surface as a confusing worker-side decode failure far
            // away from the real cause, and the inline path above
            // already propagates the same error via `map_err`.
            let item_msgpack = rmp_serde::to_vec_named(&item_value)
                .map_err(|e| format!("msgpack encode offloaded item: {}", e))?;
            offload_key = format!("{}_{}.bin", shared.request_id, index);
            // Track the request as offloaded BEFORE the store write: this future
            // can be dropped mid-`put` when a sibling item's publish fails under
            // `try_join_all`, yet the object-store write may still have landed.
            // Recording the id first keeps the blob eligible for cleanup on the
            // error/timeout unwind (issue #1471 review).
            record_offloaded_payload_key(
                &self.offloaded_payload_keys,
                shared.request_id,
                &offload_key,
            );
            if let Err(e) = self.payload_store.put(&offload_key, &item_msgpack).await {
                warn!(error = %e, "failed to offload payload, sending inline");
            } else {
                ref_item.item = None;
                ref_item.payload_ref = Some(&offload_key);
                encoded = rmp_serde::to_vec_named(&ref_item)
                    .map_err(|e| format!("msgpack encode offloaded: {}", e))?;
                metrics::QUEUE_PAYLOAD_OFFLOADS.inc();
            }
        }

        let ack = self
            .jetstream
            .publish(subject.to_string(), encoded.clone().into())
            .await
            .map_err(|e| format!("publish work item {}/{}: {}", index, total_items, e))?;
        Ok((ack, encoded))
    }

    /// Handle an incoming result message (called from inbox subscription).
    pub async fn handle_result(&self, result: WorkResult) {
        let request_id = result.request_id.clone();

        // Insert result into collector (per-key lock via DashMap)
        let all_done = {
            let mut entry = match self.pending_results.get_mut(&request_id) {
                Some(e) => e,
                None => {
                    warn!(request_id = %request_id, "received result for unknown request");
                    return;
                }
            };
            let collector = entry.value_mut();
            let idx = result.item_index as usize;
            if idx < collector.results.len() {
                collector.results[idx] = Some(result);
            }
            collector.results.iter().all(|r| r.is_some())
        };
        // DashMap per-key lock is released here

        if all_done {
            // Remove atomically — only one thread can win this remove
            if let Some((_, mut collector)) = self.pending_results.remove(&request_id) {
                let operation = collector.operation.clone();
                let wait_secs = collector.published_at.elapsed().as_secs_f64();
                let results: Vec<WorkResult> =
                    collector.results.drain(..).map(|r| r.unwrap()).collect();

                if let Some(sender) = collector.sender.take() {
                    let _ = sender.send(results);
                }

                metrics::QUEUE_RESULT_WAIT
                    .with_label_values(&[&operation])
                    .observe(wait_secs);

                self.cleanup_offloaded_payloads(&request_id).await;
            }
        }
    }

    /// Start the inbox subscription for result collection.
    /// Aborts the previous inbox loop if one exists (prevents duplicates on NATS reconnect).
    pub async fn start_inbox_subscription(
        self: &Arc<Self>,
        client: &async_nats::Client,
    ) -> Result<(), String> {
        // Stash the client so the cancel path can publish
        // ``cancel.{router_id}.{request_id}`` without re-acquiring a
        // handle from the nats manager.
        {
            let mut slot = self.nats_client.write().await;
            *slot = Some(client.clone());
        }

        let inbox_subject = format!("_INBOX.{}.>", self.router_id);

        let subscriber = client
            .subscribe(inbox_subject.clone())
            .await
            .map_err(|e| format!("subscribe inbox: {}", e))?;

        let publisher = Arc::clone(self);
        let new_handle = tokio::spawn(async move {
            publisher.handle_inbox(subscriber).await;
        });

        // Abort previous inbox loop before storing the new handle
        let mut handle_guard = self.inbox_handle.lock().await;
        if let Some(old_handle) = handle_guard.take() {
            old_handle.abort();
            debug!("aborted previous inbox subscription");
        }
        *handle_guard = Some(new_handle);

        info!(subject = %inbox_subject, "inbox subscription started");
        Ok(())
    }

    async fn handle_inbox(&self, mut subscriber: async_nats::Subscriber) {
        while let Some(msg) = subscriber.next().await {
            // Fast-path: extract request_id without full deserialization.
            // DashMap contains_key is lock-free.
            if let Some(request_id) = extract_request_id_fast(&msg.payload) {
                if !self.pending_results.contains_key(request_id)
                    && !self.pending_streams.contains_key(request_id)
                {
                    debug!(
                        request_id = %request_id,
                        "fast-path skip: result for unknown request"
                    );
                    metrics::QUEUE_INBOX_SKIPS.inc();
                    continue;
                }
            }

            // Decode the msgpack payload exactly once into an
            // `rmpv::Value`, peek the `kind` discriminator on the decoded
            // value, then convert the *same* value into the typed
            // envelope via `rmpv::ext::from_value` — no second decode
            // from the raw slice. The previous code re-decoded the
            // payload up to three times per chunk (`is_chunk_envelope`
            // peek + `is_nak_envelope` peek + the typed `from_slice`),
            // which is the hot path for streaming generation.
            let value: rmpv::Value = match rmp_serde::from_slice(&msg.payload) {
                Ok(v) => v,
                Err(e) => {
                    warn!(error = %e, "failed to decode inbox payload");
                    continue;
                }
            };

            match envelope_kind(&value) {
                // Chunk envelopes feed the streaming aggregator.
                Some("chunk") => match rmpv::ext::from_value::<ChunkEnvelope>(value) {
                    Ok(chunk) => self.handle_chunk(chunk).await,
                    Err(e) => warn!(error = %e, "failed to decode chunk envelope"),
                },
                // Worker-emitted NAK. The publisher republishes the
                // cached work item to the pool subject; if the republish
                // fails we surface a transport-style terminal outcome to
                // the HTTP handler.
                Some("nak") => match rmpv::ext::from_value::<NakEnvelope>(value) {
                    Ok(nak) => self.handle_nak(nak).await,
                    Err(e) => warn!(error = %e, "failed to decode nak envelope"),
                },
                // Anything else (encode/score/extract WorkResults — an
                // array-shaped payload, or a map with an unknown/absent
                // `kind`) flows to the ResultCollector path.
                _ => match rmpv::ext::from_value::<WorkResult>(value) {
                    Ok(result) => self.handle_result(result).await,
                    Err(e) => warn!(error = %e, "failed to decode inbox result"),
                },
            }
        }

        warn!("inbox subscription ended");
    }

    /// Handle a worker-emitted NAK envelope.
    ///
    /// Maps the envelope's ``reason`` field into one of a small
    /// closed set of metric labels so dashboards can distinguish
    /// the failure mode (KV budget vs. model-not-loaded vs.
    /// worker-shutting-down) without unbounded cardinality.
    /// Unknown reasons fall through to the generic ``nak`` bucket so
    /// we never lose data on a forward-compat addition.
    async fn handle_nak(&self, nak: NakEnvelope) {
        let reason: &'static str = match nak.reason.as_str() {
            "kv_budget" => "nak_kv_budget",
            "model_not_loaded" => "nak_model_not_loaded",
            "worker_shutting_down" => "nak_worker_shutting_down",
            other => {
                // Surface unknown reasons via a warn so a future worker
                // adding a new reason without a matching gateway update is
                // visible in logs. Production degrades gracefully via the
                // `"nak"` catch-all bucket (bounded cardinality). We do
                // NOT `debug_assert!` here: a newer worker emitting a
                // not-yet-known reason is a legitimate forward-compat
                // scenario, and the assert would crash debug/test builds
                // for any non-empty unknown reason rather than degrade.
                tracing::warn!(
                    request_id = %nak.request_id,
                    reason = %other,
                    "unknown NAK reason — bucketing as `nak`"
                );
                "nak"
            }
        };
        match self
            .republish_to_pool_outcome(&nak.request_id, reason)
            .await
        {
            Ok(RepublishOutcome::Republished) => debug!(
                request_id = %nak.request_id,
                reason = %nak.reason,
                "NAK observed, republished to pool"
            ),
            Ok(RepublishOutcome::AlreadyRepublished) => {
                // The request already fell back to the pool (via an
                // earlier NAK or the first-chunk timeout) and is now
                // being NAKed again. There is nothing further to retry,
                // so don't leave the client hanging until the first-chunk
                // timeout — surface a 429 immediately, mirroring the
                // pool-republish-failed arm below.
                let (display_model, pool) = self
                    .pending_streams
                    .get(&nak.request_id)
                    .map(|e| (e.value().display_model.clone(), e.value().pool.clone()))
                    .unwrap_or_else(|| ("unknown".to_string(), "unknown".to_string()));
                metrics::RATE_LIMIT_TOTAL
                    .with_label_values(&[
                        &metrics::sanitize_model_label(&display_model),
                        &metrics::sanitize_label(&pool),
                        "kv_pool_saturated",
                    ])
                    .inc();
                warn!(
                    request_id = %nak.request_id,
                    reason = %nak.reason,
                    "NAK on already-republished request — surfacing 429 to client"
                );
                self.fail_pending_stream(
                    &nak.request_id,
                    "rate_limit_exceeded",
                    "KV cache saturated and request already retried on the pool",
                );
            }
            Ok(RepublishOutcome::NotPossible) => debug!(
                request_id = %nak.request_id,
                "NAK observed but no republish performed (no collector or payload)"
            ),
            Ok(RepublishOutcome::RateLimited) => {
                // H9 — only the `first_chunk_timeout` reason path
                // exercises the bucket; the NAK path always supplies
                // a NAK-reason string and never trips this arm. Log
                // defensively in case a future refactor routes a NAK
                // through the bucket, so the request isn't silently
                // stranded.
                warn!(
                    request_id = %nak.request_id,
                    reason = %nak.reason,
                    "NAK republish unexpectedly rate-limited — surfacing 429 to client"
                );
                self.fail_pending_stream(
                    &nak.request_id,
                    "rate_limit_exceeded",
                    "fallback rate limit reached during NAK republish",
                );
            }
            Err(e) => {
                // Pool republish failed for an already-NAKed request:
                // both the direct-dispatched worker and the pool target
                // are unable to service the request. Surface a typed
                // ``rate_limit_exceeded`` outcome so the HTTP handler
                // returns 429 + Retry-After immediately instead of
                // waiting out the first-chunk timeout.
                // Fall back to ``unknown`` labels rather than the
                // empty string when the entry was already torn down by
                // a concurrent terminal — avoids polluting the metric
                // with empty label series that don't render in
                // dashboards.
                let (display_model, pool) = self
                    .pending_streams
                    .get(&nak.request_id)
                    .map(|e| (e.value().display_model.clone(), e.value().pool.clone()))
                    .unwrap_or_else(|| ("unknown".to_string(), "unknown".to_string()));
                metrics::RATE_LIMIT_TOTAL
                    .with_label_values(&[
                        &metrics::sanitize_model_label(&display_model),
                        &metrics::sanitize_label(&pool),
                        "kv_pool_saturated",
                    ])
                    .inc();
                warn!(
                    error = %e,
                    request_id = %nak.request_id,
                    reason = %nak.reason,
                    "NAK republish to pool failed — surfacing 429 to client"
                );
                self.fail_pending_stream(
                    &nak.request_id,
                    "rate_limit_exceeded",
                    "KV cache saturated and pool republish failed",
                );
            }
        }
    }

    /// Apply a streaming chunk envelope to the per-request collector.
    /// On terminal chunks, fire the outcome sender and remove the entry.
    async fn handle_chunk(&self, chunk: ChunkEnvelope) {
        let request_id = chunk.request_id.clone();
        let applied = {
            let mut entry = match self.pending_streams.get_mut(&request_id) {
                Some(e) => e,
                None => {
                    debug!(request_id = %request_id, "stream chunk for unknown request");
                    return;
                }
            };
            entry.value_mut().apply(chunk)
        };

        match applied {
            ChunkApplied::Terminal => {
                if let Some((_, mut collector)) = self.pending_streams.remove(&request_id) {
                    let wait_secs = collector.published_at.elapsed().as_secs_f64();
                    let outcome = collector.build_outcome();
                    if let (Some(sender), Some(outcome)) = (collector.sender.take(), outcome) {
                        if sender.send(outcome).is_err() {
                            debug!(
                                request_id = %request_id,
                                "terminal outcome receiver dropped (client likely disconnected)"
                            );
                        }
                    }
                    metrics::QUEUE_RESULT_WAIT
                        .with_label_values(&["generate"])
                        .observe(wait_secs);
                    // Stream finished — drop any offloaded generate blob now
                    // (the periodic reconcile is only the failure-path backstop).
                    self.cleanup_offloaded_generate(&request_id).await;
                }
            }
            ChunkApplied::SeqGap => {
                // H6: a per-attempt seq gap means a required content
                // chunk was lost on the worker → gateway transport.
                // Mirror the worker's no-silent-drop guarantee on the
                // gateway side: fail the pending stream with
                // ``transport_failure`` so the client sees an explicit
                // error rather than a silently shortened completion.
                warn!(
                    request_id = %request_id,
                    "streaming seq gap detected — failing pending stream as transport_failure"
                );
                self.fail_pending_stream(
                    &request_id,
                    "transport_failure",
                    "streaming chunk sequence gap (missing chunk between worker and gateway)",
                );
            }
            ChunkApplied::Delta | ChunkApplied::Stale | ChunkApplied::Duplicate => {}
        }
    }

    /// Drain pending result collectors on graceful shutdown.
    /// Waits up to `timeout` for in-flight results to arrive, then drops the rest.
    pub async fn drain_pending(&self, timeout: Duration) {
        let deadline = Instant::now() + timeout;
        let poll_interval = Duration::from_millis(100);

        loop {
            if self.pending_results.is_empty() && self.pending_streams.is_empty() {
                info!("all pending queue results drained");
                return;
            }
            let count = self.pending_results.len() + self.pending_streams.len();
            if Instant::now() >= deadline {
                warn!(
                    remaining = count,
                    "shutdown drain timeout — dropping pending results"
                );
                break;
            }
            debug!(
                remaining = count,
                "waiting for pending queue results to drain"
            );
            tokio::time::sleep(poll_interval).await;
        }

        // Force-complete remaining collectors so senders don't leak
        self.cleanup_expired().await;
        self.cleanup_pending_streams().await;
    }

    /// Clean up expired result collectors.
    pub async fn cleanup_expired(&self) {
        let now = Instant::now();

        // Reconcile retained payload-delete failures for completed/failed
        // non-streaming requests. Live requests stay protected by their
        // ``pending_results`` entry; once that entry is gone, only the exact
        // recorded object-store keys are retried.
        let orphaned_payloads = collect_orphaned_offloaded_payload_requests(
            &self.offloaded_payload_keys,
            |request_id| self.pending_results.contains_key(request_id),
        );
        for request_id in orphaned_payloads {
            self.cleanup_offloaded_payloads(&request_id).await;
        }

        let expired: Vec<String> = self
            .pending_results
            .iter()
            .filter(|entry| now > entry.value().deadline)
            .map(|entry| entry.key().clone())
            .collect();

        for key in &expired {
            if let Some((_, mut collector)) = self.pending_results.remove(key) {
                warn!(request_id = %key, "result collector timed out");
                let results: Vec<WorkResult> = collector.results.drain(..).flatten().collect();
                if let Some(sender) = collector.sender.take() {
                    if sender.send(results).is_err() {
                        debug!(
                            request_id = %key,
                            "expired-result receiver dropped (client likely disconnected)"
                        );
                    }
                }
            }
            self.cleanup_offloaded_payloads(key).await;
        }

        // Reconcile offloaded generate blobs: any tracked request whose stream
        // is no longer pending has terminated (success cleans up at the
        // terminal funnel; this is the backstop for failure/timeout paths,
        // including the sync ``fail_pending_stream`` which can't await).
        //
        // Check BOTH in-flight maps: ``publish_generate`` is shared with the
        // batch generate arm (``publish_work``), which tracks requests in
        // ``pending_results`` rather than ``pending_streams``. Excluding only
        // ``pending_streams`` would let this sweep delete a batch request's
        // offloaded blob mid-flight and break redelivery. (That arm is dead
        // today — all generate goes through the streaming path — but this keeps
        // the reconcile correct if it's ever wired up.)
        let orphaned: Vec<String> = self
            .offloaded_streams
            .iter()
            .filter(|id| {
                !self.pending_streams.contains_key(id.key())
                    && !self.pending_results.contains_key(id.key())
            })
            .map(|id| id.key().clone())
            .collect();
        for request_id in orphaned {
            self.cleanup_offloaded_generate(&request_id).await;
        }
    }

    /// Delete an offloaded generate blob (``{request_id}_0.bin``) and drop its
    /// tracking entry. Best-effort + idempotent: a no-op when the request had
    /// no offload, so it is safe to call on every stream termination.
    async fn cleanup_offloaded_generate(&self, request_id: &str) {
        // No-op for non-offloaded requests.
        if !self.offloaded_streams.contains(request_id) {
            return;
        }
        let key = format!("{request_id}_0.bin");
        // Delete the blob FIRST and only drop the tracker on success, so a
        // transient delete failure leaves the entry for the next periodic
        // reconcile to retry instead of leaking the blob.
        match self.payload_store.delete(&key).await {
            Ok(()) => {
                self.offloaded_streams.remove(request_id);
            }
            Err(e) => {
                warn!(
                    key = %key,
                    error = %e,
                    "failed to delete offloaded generate payload; will retry on next reconcile"
                );
            }
        }
    }

    async fn cleanup_pending_streams(&self) {
        let expired: Vec<String> = self
            .pending_streams
            .iter()
            .map(|entry| entry.key().clone())
            .collect();

        for key in expired {
            if let Some((_, mut collector)) = self.pending_streams.remove(&key) {
                warn!(request_id = %key, "stream collector timed out");
                let wait_secs = collector.published_at.elapsed().as_secs_f64();
                let outcome = collector.build_outcome().unwrap_or_else(|| StreamOutcome {
                    text: String::new(),
                    finish_reason: "error".to_string(),
                    usage: None,
                    attempt_id: collector.current_attempt_id.clone().unwrap_or_default(),
                    ttft_ms: None,
                    tpot_ms: None,
                    error: Some(ChunkError {
                        code: "shutdown".to_string(),
                        message: "gateway shutdown before stream completed".to_string(),
                    }),
                    tool_calls: None,
                    logprobs: None,
                    candidates: Vec::new(),
                });
                if let Some(sender) = collector.sender.take() {
                    if sender.send(outcome).is_err() {
                        debug!(
                            request_id = %key,
                            "shutdown-drain receiver dropped (client likely disconnected)"
                        );
                    }
                }
                metrics::QUEUE_RESULT_WAIT
                    .with_label_values(&["generate"])
                    .observe(wait_secs);
                // Drop any offloaded blob for this shutdown-dropped stream. The
                // periodic reconcile filters against ``pending_streams``, so an
                // entry removed here would otherwise be missed at shutdown.
                self.cleanup_offloaded_generate(&key).await;
            }
        }
    }

    /// Remove offloaded payloads for a completed/expired request. No-op when the
    /// request offloaded nothing, and exact-key cleanup for mixed requests that
    /// offloaded only a subset of possible payload blobs.
    async fn cleanup_offloaded_payloads(&self, request_id: &str) {
        cleanup_offloaded_payloads_inner(
            self.payload_store.as_ref(),
            &self.offloaded_payload_keys,
            request_id,
        )
        .await;
    }
}

fn collect_orphaned_offloaded_payload_requests(
    offloaded: &DashMap<String, BTreeSet<String>>,
    is_pending: impl Fn(&str) -> bool,
) -> Vec<String> {
    offloaded
        .iter()
        .filter(|entry| !is_pending(entry.key()))
        .map(|entry| entry.key().clone())
        .collect()
}

fn record_offloaded_payload_key(
    offloaded: &DashMap<String, BTreeSet<String>>,
    request_id: &str,
    key: &str,
) {
    offloaded
        .entry(request_id.to_string())
        .or_default()
        .insert(key.to_string());
}

/// Delete the object-store blobs an offloaded request wrote or attempted. No-op
/// when the request offloaded nothing. Successful deletes are dropped from
/// tracking; failed deletes are retained so the periodic reconcile can retry.
/// Extracted from [`WorkPublisher::cleanup_offloaded_payloads`] so the guard is
/// unit-testable without a live JetStream context (issue #1471).
async fn cleanup_offloaded_payloads_inner(
    payload_store: &dyn PayloadStore,
    offloaded: &DashMap<String, BTreeSet<String>>,
    request_id: &str,
) {
    let Some((_, keys)) = offloaded.remove(request_id) else {
        return;
    };

    let mut failed = BTreeSet::new();
    for key in keys {
        if let Err(e) = payload_store.delete(&key).await {
            warn!(
                key = %key,
                error = %e,
                "failed to remove offloaded payload; will retry on next reconcile"
            );
            failed.insert(key);
        }
    }

    if !failed.is_empty() {
        offloaded
            .entry(request_id.to_string())
            .or_default()
            .extend(failed);
    }
}

/// Content negotiation: determine if client wants msgpack or JSON.
pub fn wants_msgpack(headers: &axum::http::HeaderMap) -> bool {
    headers
        .get("accept")
        .and_then(|v| v.to_str().ok())
        .map(|accept| {
            accept.contains("application/msgpack")
                || accept.contains("application/x-msgpack")
                || accept.contains("application/vnd.msgpack")
        })
        .unwrap_or(false)
}

/// Serialize response based on content negotiation.
#[allow(dead_code)]
pub fn encode_response(
    data: &impl Serialize,
    use_msgpack: bool,
) -> Result<(String, Vec<u8>), String> {
    if use_msgpack {
        let bytes = rmp_serde::to_vec(data).map_err(|e| format!("msgpack encode: {}", e))?;
        Ok(("application/msgpack".to_string(), bytes))
    } else {
        let bytes = serde_json::to_vec(data).map_err(|e| format!("json encode: {}", e))?;
        Ok(("application/json".to_string(), bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- issue #1471: payload-store cleanup guard ---
    use async_trait::async_trait;

    /// Records every `delete` so tests can assert the cleanup guard issues no
    /// object-store DELETEs for inline (non-offloaded) requests.
    #[derive(Default)]
    struct CountingPayloadStore {
        deleted: std::sync::Mutex<Vec<String>>,
        fail_delete: bool,
    }

    #[async_trait]
    impl PayloadStore for CountingPayloadStore {
        async fn put(&self, _key: &str, _data: &[u8]) -> Result<String, String> {
            Ok(String::new())
        }
        async fn delete(&self, key: &str) -> Result<(), String> {
            self.deleted.lock().unwrap().push(key.to_string());
            if self.fail_delete {
                Err("simulated object-store delete failure".to_string())
            } else {
                Ok(())
            }
        }
    }

    // Common case (issue #1471): a small/inline request that never offloaded a
    // payload must issue zero object-store DELETEs. The unconditional
    // per-request DELETE storm is exactly what capped throughput.
    #[tokio::test]
    async fn cleanup_offloaded_payloads_noop_when_nothing_offloaded() {
        let store = CountingPayloadStore::default();
        let offloaded: DashMap<String, BTreeSet<String>> = DashMap::new();
        cleanup_offloaded_payloads_inner(&store, &offloaded, "req-inline").await;
        let deleted = store.deleted.lock().unwrap();
        assert!(
            deleted.is_empty(),
            "inline request must not trigger payload-store deletes, got {:?}",
            *deleted,
        );
    }

    #[test]
    fn orphaned_offloaded_payload_requests_skip_live_pending_results() {
        let offloaded: DashMap<String, BTreeSet<String>> = DashMap::new();
        record_offloaded_payload_key(&offloaded, "req-done", "req-done_0.bin");
        record_offloaded_payload_key(&offloaded, "req-live", "req-live_0.bin");
        let got = collect_orphaned_offloaded_payload_requests(&offloaded, |request_id| {
            request_id == "req-live"
        });
        assert_eq!(
            got,
            vec!["req-done".to_string()],
            "reconcile must retry only tracked payloads whose request is no longer pending",
        );
    }

    // A request that actually offloaded deletes only the concrete keys that
    // were recorded. This keeps mixed batches from deleting every possible
    // `{id}_{i}.bin` sibling just because one item crossed the threshold.
    #[tokio::test]
    async fn cleanup_offloaded_payloads_deletes_only_tracked_keys() {
        let store = CountingPayloadStore::default();
        let offloaded: DashMap<String, BTreeSet<String>> = DashMap::new();
        record_offloaded_payload_key(&offloaded, "req-big", "req-big_1.bin");
        record_offloaded_payload_key(&offloaded, "req-big", "req-big_score.bin");
        cleanup_offloaded_payloads_inner(&store, &offloaded, "req-big").await;
        let mut got = store.deleted.lock().unwrap().clone();
        got.sort();
        assert_eq!(
            got,
            vec!["req-big_1.bin".to_string(), "req-big_score.bin".to_string(),],
            "cleanup must delete only the keys that were actually recorded",
        );
        assert!(
            !offloaded.contains_key("req-big"),
            "tracking entry must be dropped after cleanup",
        );
    }

    // A score request records only the score blob, so cleanup must not infer an
    // item blob from the request id.
    #[tokio::test]
    async fn cleanup_offloaded_payloads_score_key_only_when_tracked() {
        let store = CountingPayloadStore::default();
        let offloaded: DashMap<String, BTreeSet<String>> = DashMap::new();
        record_offloaded_payload_key(&offloaded, "req-score", "req-score_score.bin");
        cleanup_offloaded_payloads_inner(&store, &offloaded, "req-score").await;
        let got = store.deleted.lock().unwrap().clone();
        assert_eq!(
            got,
            vec!["req-score_score.bin".to_string()],
            "score cleanup must delete only the tracked score key",
        );
        assert!(!offloaded.contains_key("req-score"));
    }

    // Cleanup is idempotent after a successful cleanup: a second call (both
    // terminal cleanup paths can fire) is a no-op once the first dropped the
    // tracking entry.
    #[tokio::test]
    async fn cleanup_offloaded_payloads_second_call_is_noop() {
        let store = CountingPayloadStore::default();
        let offloaded: DashMap<String, BTreeSet<String>> = DashMap::new();
        record_offloaded_payload_key(&offloaded, "req-twice", "req-twice_0.bin");
        cleanup_offloaded_payloads_inner(&store, &offloaded, "req-twice").await;
        let after_first = store.deleted.lock().unwrap().len();
        cleanup_offloaded_payloads_inner(&store, &offloaded, "req-twice").await;
        let after_second = store.deleted.lock().unwrap().len();
        assert_eq!(
            after_first, after_second,
            "second cleanup must issue no further deletes (entry already removed)",
        );
    }

    // Direct-publish fallback can attempt the same offload key more than once
    // for the same request. Track unique keys so cleanup remains O(unique
    // object keys), not O(publish attempts).
    #[tokio::test]
    async fn cleanup_offloaded_payloads_deduplicates_republish_keys() {
        let store = CountingPayloadStore::default();
        let offloaded: DashMap<String, BTreeSet<String>> = DashMap::new();
        record_offloaded_payload_key(&offloaded, "req-retry", "req-retry_0.bin");
        record_offloaded_payload_key(&offloaded, "req-retry", "req-retry_0.bin");
        cleanup_offloaded_payloads_inner(&store, &offloaded, "req-retry").await;
        let got = store.deleted.lock().unwrap().clone();
        assert_eq!(
            got,
            vec!["req-retry_0.bin".to_string()],
            "duplicate key records must produce one payload-store delete",
        );
    }

    // A failing object store must not break cleanup: every delete is attempted,
    // failed keys remain tracked, and a later successful pass drains them.
    #[tokio::test]
    async fn cleanup_offloaded_payloads_retains_failed_deletes_for_retry() {
        let store = CountingPayloadStore {
            fail_delete: true,
            ..Default::default()
        };
        let offloaded: DashMap<String, BTreeSet<String>> = DashMap::new();
        record_offloaded_payload_key(&offloaded, "req-fail", "req-fail_1.bin");
        record_offloaded_payload_key(&offloaded, "req-fail", "req-fail_score.bin");
        cleanup_offloaded_payloads_inner(&store, &offloaded, "req-fail").await;
        let mut attempted = store.deleted.lock().unwrap().clone();
        attempted.sort();
        assert_eq!(
            attempted,
            vec![
                "req-fail_1.bin".to_string(),
                "req-fail_score.bin".to_string(),
            ],
            "all deletes must be attempted even when the store errors",
        );
        {
            let retained = offloaded.get("req-fail").expect("failed keys retained");
            assert_eq!(
                retained.iter().cloned().collect::<Vec<_>>(),
                vec![
                    "req-fail_1.bin".to_string(),
                    "req-fail_score.bin".to_string(),
                ],
                "failed delete keys must remain eligible for retry",
            );
        }

        let retry_store = CountingPayloadStore::default();
        cleanup_offloaded_payloads_inner(&retry_store, &offloaded, "req-fail").await;
        let mut retry_attempted = retry_store.deleted.lock().unwrap().clone();
        retry_attempted.sort();
        assert_eq!(
            retry_attempted,
            vec![
                "req-fail_1.bin".to_string(),
                "req-fail_score.bin".to_string(),
            ],
            "retry cleanup must attempt retained keys",
        );
        assert!(!offloaded.contains_key("req-fail"));
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct WorkItem {
        pub work_item_id: String,
        pub request_id: String,
        pub item_index: u32,
        pub total_items: u32,
        pub operation: String,
        pub model_id: String,
        #[serde(default)]
        pub profile_id: String,
        #[serde(default)]
        pub engine: String,
        pub pool_name: String,
        #[serde(default, skip_serializing_if = "String::is_empty")]
        pub admission_pool: String,
        pub machine_profile: String,
        #[serde(default)]
        pub item: Option<rmpv::Value>,
        #[serde(default)]
        pub payload_ref: Option<String>,
        #[serde(default)]
        pub output_types: Option<Vec<String>>,
        #[serde(default)]
        pub instruction: Option<String>,
        #[serde(default)]
        pub is_query: bool,
        #[serde(default)]
        pub options: Option<serde_json::Value>,
        #[serde(default)]
        pub query_item: Option<rmpv::Value>,
        #[serde(default)]
        pub query_payload_ref: Option<String>,
        #[serde(default)]
        pub score_items: Option<Vec<rmpv::Value>>,
        #[serde(default)]
        pub labels: Option<Vec<String>>,
        #[serde(default)]
        pub output_schema: Option<serde_json::Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub generate: Option<GenerateParams>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub routing_key: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub prompt_cache_key: Option<String>,
        #[serde(default)]
        pub bundle_config_hash: String,
        #[serde(default)]
        pub router_id: String,
        pub reply_subject: String,
        #[serde(default)]
        pub timestamp: f64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub traceparent: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub tracestate: Option<String>,
    }

    #[test]
    fn test_stream_name() {
        assert_eq!(stream_name("default"), "WORK_POOL_default");
        assert_eq!(stream_name("eval-l4"), "WORK_POOL_eval-l4");
    }

    #[test]
    fn test_canonical_stream_subjects_replaces_legacy_subject() {
        let observed = vec!["sie.work.*.default".to_string()];
        assert_eq!(
            canonical_stream_subjects(observed, "sie.work.default.*.*.*"),
            Some(vec!["sie.work.default.*.*.*".to_string()])
        );
    }

    #[test]
    fn test_canonical_stream_subjects_drops_extra_subjects() {
        let observed = vec![
            "sie.work.*.default".to_string(),
            "sie.work.other.*.*.*".to_string(),
        ];
        assert_eq!(
            canonical_stream_subjects(observed, "sie.work.default.*.*.*"),
            Some(vec!["sie.work.default.*.*.*".to_string()])
        );
    }

    #[test]
    fn test_canonical_stream_subjects_noops_when_exact() {
        let observed = vec!["sie.work.default.*.*.*".to_string()];
        assert_eq!(
            canonical_stream_subjects(observed, "sie.work.default.*.*.*"),
            None
        );
    }

    #[test]
    fn test_direct_batch_fallback_payloads_only_include_missing_items_once() {
        let (tx, _rx) = oneshot::channel();
        let mut collector = ResultCollector {
            _total_items: 3,
            results: vec![
                None,
                Some(WorkResult {
                    work_item_id: "req.1".into(),
                    request_id: "req".into(),
                    item_index: 1,
                    success: true,
                    result_msgpack: Vec::new(),
                    error: None,
                    error_code: None,
                    inference_ms: None,
                    queue_ms: None,
                    processing_ms: None,
                    worker_id: None,
                    tokenization_ms: None,
                    postprocessing_ms: None,
                    payload_fetch_ms: None,
                }),
                None,
            ],
            sender: Some(tx),
            deadline: Instant::now() + Duration::from_secs(30),
            operation: "encode".into(),
            published_at: Instant::now(),
            pool_label: "default".into(),
            model_label: "BAAI/bge-m3".into(),
            pool_fallback_subject: Some("sie.work.default.l4.default.BAAI__bge-m3".into()),
            direct_fallback_payloads: vec![Some(vec![0]), Some(vec![1]), Some(vec![2])],
            direct_fallback_republished: false,
        };

        let (subject, payloads) =
            take_direct_fallback_payloads(&mut collector).expect("fallback payloads");
        assert_eq!(subject, "sie.work.default.l4.default.BAAI__bge-m3");
        assert_eq!(payloads, vec![(0, vec![0]), (2, vec![2])]);
        assert!(collector.direct_fallback_republished);
        assert!(take_direct_fallback_payloads(&mut collector).is_none());
    }

    #[test]
    fn test_pending_generation_snapshot_groups_collectors() {
        let now = Instant::now();
        let mut grouped: BTreeMap<(String, String, String), PendingGenerationGroup> =
            BTreeMap::new();

        let (tx1, _rx1) = oneshot::channel();
        let mut waiting = StreamCollector::new(tx1, "model-a:no-spec".into(), "default".into());
        waiting.display_model = "model-a".into();
        waiting.published_at = now.checked_sub(Duration::from_millis(250)).unwrap_or(now);
        accumulate_pending_generation_group(&mut grouped, &waiting, now);

        let (tx2, _rx2) = oneshot::channel();
        let mut active = StreamCollector::new(tx2, "model-a:no-spec".into(), "default".into());
        active.display_model = "model-a".into();
        active.first_chunk_at = Some(now);
        active.republished = true;
        active.published_at = now.checked_sub(Duration::from_millis(100)).unwrap_or(now);
        accumulate_pending_generation_group(&mut grouped, &active, now);

        let (tx3, _rx3) = oneshot::channel();
        let other = StreamCollector::new(tx3, "model-b".into(), "l4".into());
        accumulate_pending_generation_group(&mut grouped, &other, now);

        let groups: Vec<PendingGenerationGroup> = grouped.into_values().collect();
        let snapshot = PendingGenerationSnapshot {
            total: groups.iter().map(|group| group.count).sum(),
            groups,
        };

        assert_eq!(snapshot.total, 3);
        let model_a = snapshot.for_model("model-a");
        assert_eq!(model_a.total, 2);
        assert_eq!(model_a.groups.len(), 1);
        assert_eq!(model_a.groups[0].count, 2);
        assert_eq!(model_a.groups[0].waiting_first_chunk, 1);
        assert_eq!(model_a.groups[0].active_streams, 1);
        assert_eq!(model_a.groups[0].republished, 1);
        assert!(model_a.groups[0].oldest_request_age_ms >= 250);

        let model_b = snapshot.for_model("model-b");
        assert_eq!(model_b.total, 1);
        assert_eq!(model_b.groups[0].pool, "l4");
    }

    #[test]
    fn test_normalize_model_id() {
        assert_eq!(normalize_model_id("BAAI/bge-m3"), "BAAI__bge-m3");
        assert_eq!(normalize_model_id("my-model"), "my-model");
        assert_eq!(
            normalize_model_id("vidore/colqwen2.5-v0.2"),
            "vidore__colqwen2_dot_5-v0_dot_2"
        );
        assert_eq!(
            normalize_model_id("sentence-transformers/all-MiniLM-L6-v2"),
            "sentence-transformers__all-MiniLM-L6-v2"
        );
        assert_eq!(normalize_model_id("a*b"), "a_b");
        assert_eq!(normalize_model_id("a>b"), "a_b");
        assert_eq!(normalize_model_id("a b"), "a_b");
    }

    /// Cross-language fixture for the worker_id normalization contract.
    ///
    /// Mirrors `packages/sie_sdk/tests/test_worker_id_normalization.py` —
    /// both languages share the same NATS subject and the two sides MUST
    /// produce byte-identical tokens or direct-dispatch silently misses
    /// (workstream G-M5).
    ///
    /// Worker IDs flow through `normalize_model_id` (same function as the
    /// model id, since both targets are single NATS subject tokens with
    /// the same legality rules). Case is **preserved** intentionally —
    /// NATS subjects are case-sensitive and operator-set worker names
    /// commonly include mixed case; lowercasing would surprise operators
    /// and force a cross-language migration with no benefit.
    #[test]
    fn test_worker_id_normalization_cross_language() {
        // unchanged: clean ascii with hyphens
        assert_eq!(normalize_model_id("worker-1"), "worker-1");
        // case preserved (NOT lowercased — see doc comment above)
        assert_eq!(normalize_model_id("Worker-1"), "Worker-1");
        assert_eq!(normalize_model_id("WORKER"), "WORKER");
        // dotted Kubernetes pod hostname → each dot → "_dot_"
        assert_eq!(
            normalize_model_id("sie-worker-7d9f-default-0.sie-worker.default.svc"),
            "sie-worker-7d9f-default-0_dot_sie-worker_dot_default_dot_svc"
        );
        // whitespace → "_"
        assert_eq!(normalize_model_id("my worker"), "my_worker");
        // wildcard tokens → "_". Each `.` → `_dot_` and `*` → `_`, so
        // `worker.*.foo` = `worker` + `_dot_` + `_` + `_dot_` + `foo`.
        assert_eq!(normalize_model_id("worker.*.foo"), "worker_dot___dot_foo");
        // leading/trailing whitespace is preserved as `_` (we do NOT trim —
        // Python helper rejects whitespace-only ids upstream; non-empty
        // padding is mapped through the same scrub).
        assert_eq!(normalize_model_id("  worker-1  "), "__worker-1__");
        // consecutive separators are NOT collapsed (no benefit; would
        // diverge from Python and break this fixture)
        assert_eq!(normalize_model_id("worker--1"), "worker--1");
    }

    #[test]
    fn test_work_subject() {
        assert_eq!(
            work_subject("default", "rtx6000", "default", "BAAI/bge-m3"),
            "sie.work.default.rtx6000.default.BAAI__bge-m3"
        );
        assert_eq!(
            work_subject("eval-l4", "l4-spot", "sglang", "my-model"),
            "sie.work.eval-l4.l4-spot.sglang.my-model"
        );
        assert_eq!(
            work_subject_worker(
                "default",
                "rtx6000",
                "sglang",
                "Qwen/Qwen3.6-27B",
                "worker-0.svc"
            ),
            "sie.work.default.rtx6000.sglang.Qwen__Qwen3_dot_6-27B.worker-0_dot_svc"
        );
    }

    /// Regression: model IDs containing `.` must produce exactly 6
    /// subject tokens (`sie`, `work`, `{pool}`, `{machine_profile}`,
    /// `{bundle}`, `{normalized_model}`) so the worker's consumer filter
    /// `sie.work.{pool}.{machine_profile}.{bundle}.*` matches them.
    #[test]
    fn test_work_subject_token_count_with_dotted_model() {
        let subj = work_subject("default", "l4", "default", "vidore/colqwen2.5-v0.2");
        let tokens: Vec<&str> = subj.split('.').collect();
        assert_eq!(
            tokens.len(),
            6,
            "subject {subj} must have 6 tokens to match sie.work.{{pool}}.{{machine_profile}}.{{bundle}}.*"
        );
        assert_eq!(tokens[0], "sie");
        assert_eq!(tokens[1], "work");
        assert_eq!(tokens[2], "default");
        assert_eq!(tokens[3], "l4");
        assert_eq!(tokens[4], "default");
        // Token[5] contains no '.' (it's the normalized model id).
        assert!(!tokens[5].contains('.'));
    }

    #[test]
    fn test_work_item_msgpack_roundtrip() {
        let item = WorkItem {
            work_item_id: "req-1.0".to_string(),
            request_id: "req-1".to_string(),
            item_index: 0,
            total_items: 3,
            operation: "encode".to_string(),
            model_id: "BAAI/bge-m3".to_string(),
            profile_id: String::new(),
            engine: "pytorch".to_string(),
            pool_name: "default".to_string(),
            admission_pool: "default".to_string(),
            machine_profile: "l4-spot".to_string(),
            item: Some(rmpv::Value::Map(vec![(
                rmpv::Value::from("text"),
                rmpv::Value::from("hello"),
            )])),
            payload_ref: None,
            output_types: Some(vec!["dense".to_string()]),
            instruction: None,
            is_query: false,
            options: None,
            query_item: None,
            query_payload_ref: None,
            score_items: None,
            labels: None,
            output_schema: None,
            generate: None,
            routing_key: None,
            prompt_cache_key: None,
            bundle_config_hash: "abc123".to_string(),
            router_id: "router-1".to_string(),
            reply_subject: "_INBOX.r1.req-1".to_string(),
            timestamp: 1700000000.0,
            traceparent: None,
            tracestate: None,
        };

        let encoded = rmp_serde::to_vec_named(&item).unwrap();
        let decoded: WorkItem = rmp_serde::from_slice(&encoded).unwrap();

        assert_eq!(decoded.work_item_id, "req-1.0");
        assert_eq!(decoded.request_id, "req-1");
        assert_eq!(decoded.item_index, 0);
        assert_eq!(decoded.total_items, 3);
        assert_eq!(decoded.operation, "encode");
        assert_eq!(decoded.model_id, "BAAI/bge-m3");
        assert_eq!(decoded.pool_name, "default");
        assert_eq!(decoded.admission_pool, "default");
        assert_eq!(decoded.machine_profile, "l4-spot");
        assert_eq!(
            decoded.item,
            Some(rmpv::Value::Map(vec![(
                rmpv::Value::from("text"),
                rmpv::Value::from("hello"),
            )]))
        );
        assert!(decoded.payload_ref.is_none());
        assert_eq!(decoded.output_types, Some(vec!["dense".to_string()]));
        assert_eq!(decoded.bundle_config_hash, "abc123");
        assert_eq!(decoded.router_id, "router-1");
        assert_eq!(decoded.reply_subject, "_INBOX.r1.req-1");
        assert_eq!(decoded.engine, "pytorch");
    }

    #[test]
    fn test_work_item_engine_back_compat() {
        let mut without_engine = serde_json::Map::new();
        without_engine.insert("work_item_id".into(), "req-1.0".into());
        without_engine.insert("request_id".into(), "req-1".into());
        without_engine.insert("item_index".into(), 0.into());
        without_engine.insert("total_items".into(), 1.into());
        without_engine.insert("operation".into(), "encode".into());
        without_engine.insert("model_id".into(), "m".into());
        without_engine.insert("pool_name".into(), "default".into());
        without_engine.insert("machine_profile".into(), "".into());
        without_engine.insert("reply_subject".into(), "_INBOX.x.y".into());
        let encoded = rmp_serde::to_vec_named(&without_engine).unwrap();
        let decoded: WorkItem = rmp_serde::from_slice(&encoded).unwrap();
        assert_eq!(decoded.engine, "");
        assert_eq!(decoded.operation, "encode");
    }

    /// Regression: `WorkItemRef` is the borrowed view we use on the
    /// publish hot path and it **must** serialize to the exact same
    /// msgpack bytes as the owned `WorkItem`. Any drift in field
    /// names/order/serde attrs between the two would silently break
    /// worker deserialization; lock it down here.
    #[test]
    fn test_work_item_ref_matches_owned_msgpack() {
        let owned = WorkItem {
            work_item_id: "req-ref.0".to_string(),
            request_id: "req-ref".to_string(),
            item_index: 0,
            total_items: 1,
            operation: "encode".to_string(),
            model_id: "BAAI/bge-m3".to_string(),
            profile_id: "default".to_string(),
            engine: "pytorch".to_string(),
            pool_name: "default".to_string(),
            admission_pool: "tenant".to_string(),
            machine_profile: "l4-spot".to_string(),
            item: Some(rmpv::Value::Map(vec![(
                rmpv::Value::from("text"),
                rmpv::Value::from("hello"),
            )])),
            payload_ref: None,
            output_types: Some(vec!["dense".to_string()]),
            instruction: Some("search_document".to_string()),
            is_query: false,
            options: Some(serde_json::json!({"truncate": true})),
            query_item: None,
            query_payload_ref: None,
            score_items: None,
            labels: Some(vec!["exp-a".to_string()]),
            output_schema: Some(serde_json::json!({"kind": "dense"})),
            generate: None,
            routing_key: None,
            prompt_cache_key: None,
            bundle_config_hash: "hash123".to_string(),
            router_id: "router-1".to_string(),
            reply_subject: "_INBOX.router-1.req-ref".to_string(),
            timestamp: 1_700_000_000.5,
            traceparent: None,
            tracestate: None,
        };

        let item_value = owned.item.clone().unwrap();
        let output_types = owned.output_types.clone().unwrap();
        let labels = owned.labels.clone().unwrap();
        let options = owned.options.clone().unwrap();
        let output_schema = owned.output_schema.clone().unwrap();

        let borrowed = WorkItemRef {
            work_item_id: &owned.work_item_id,
            request_id: &owned.request_id,
            item_index: owned.item_index,
            total_items: owned.total_items,
            operation: &owned.operation,
            model_id: &owned.model_id,
            profile_id: &owned.profile_id,
            engine: &owned.engine,
            pool_name: &owned.pool_name,
            admission_pool: &owned.admission_pool,
            machine_profile: &owned.machine_profile,
            item: Some(&item_value),
            payload_ref: None,
            output_types: Some(&output_types),
            instruction: owned.instruction.as_deref(),
            is_query: owned.is_query,
            options: Some(&options),
            query_item: None,
            query_payload_ref: None,
            score_items: None,
            labels: Some(&labels),
            output_schema: Some(&output_schema),
            generate: None,
            routing_key: owned.routing_key.as_deref(),
            prompt_cache_key: owned.prompt_cache_key.as_deref(),
            bundle_config_hash: &owned.bundle_config_hash,
            router_id: &owned.router_id,
            reply_subject: &owned.reply_subject,
            timestamp: owned.timestamp,
            traceparent: owned.traceparent.as_deref(),
            tracestate: owned.tracestate.as_deref(),
        };

        let owned_bytes = rmp_serde::to_vec_named(&owned).unwrap();
        let ref_bytes = rmp_serde::to_vec_named(&borrowed).unwrap();
        assert_eq!(
            ref_bytes, owned_bytes,
            "WorkItemRef must produce byte-identical msgpack to WorkItem"
        );

        // And the bytes still decode into a WorkItem cleanly.
        let decoded: WorkItem = rmp_serde::from_slice(&ref_bytes).unwrap();
        assert_eq!(decoded.work_item_id, owned.work_item_id);
        assert_eq!(decoded.item, owned.item);
        assert_eq!(decoded.options, owned.options);
    }

    #[test]
    fn test_work_result_msgpack_roundtrip() {
        let result = WorkResult {
            work_item_id: "req-1.2".to_string(),
            request_id: "req-1".to_string(),
            item_index: 2,
            success: true,
            result_msgpack: vec![5, 6, 7],
            error: None,
            error_code: None,
            inference_ms: None,
            queue_ms: None,
            processing_ms: None,
            worker_id: None,
            tokenization_ms: None,
            postprocessing_ms: None,
            payload_fetch_ms: None,
        };

        let encoded = rmp_serde::to_vec(&result).unwrap();
        let decoded: WorkResult = rmp_serde::from_slice(&encoded).unwrap();

        assert_eq!(decoded.request_id, "req-1");
        assert_eq!(decoded.item_index, 2);
        assert!(decoded.success);
        assert_eq!(decoded.result_msgpack, vec![5, 6, 7]);
    }

    #[test]
    fn test_work_item_with_payload_ref() {
        let item = WorkItem {
            work_item_id: "req-2.0".to_string(),
            request_id: "req-2".to_string(),
            item_index: 0,
            total_items: 1,
            operation: "encode".to_string(),
            model_id: "model".to_string(),
            profile_id: String::new(),
            engine: String::new(),
            pool_name: "default".to_string(),
            admission_pool: String::new(),
            machine_profile: String::new(),
            item: None,
            payload_ref: Some("/tmp/payload_req-2_0.bin".to_string()),
            output_types: None,
            instruction: None,
            is_query: false,
            options: None,
            query_item: None,
            query_payload_ref: None,
            score_items: None,
            labels: None,
            output_schema: None,
            generate: None,
            routing_key: None,
            prompt_cache_key: None,
            bundle_config_hash: String::new(),
            router_id: String::new(),
            reply_subject: "_INBOX.r1.req-2".to_string(),
            timestamp: 0.0,
            traceparent: None,
            tracestate: None,
        };

        let encoded = rmp_serde::to_vec_named(&item).unwrap();
        let decoded: WorkItem = rmp_serde::from_slice(&encoded).unwrap();

        assert!(decoded.item.is_none());
        assert_eq!(
            decoded.payload_ref,
            Some("/tmp/payload_req-2_0.bin".to_string())
        );
    }

    #[test]
    fn test_wants_msgpack() {
        let mut headers = axum::http::HeaderMap::new();
        assert!(!wants_msgpack(&headers));

        headers.insert("accept", "application/json".parse().unwrap());
        assert!(!wants_msgpack(&headers));

        headers.insert("accept", "application/msgpack".parse().unwrap());
        assert!(wants_msgpack(&headers));

        headers.insert("accept", "application/x-msgpack".parse().unwrap());
        assert!(wants_msgpack(&headers));
    }

    #[test]
    fn test_encode_response_json() {
        let data = serde_json::json!({"key": "value"});
        let (content_type, bytes) = encode_response(&data, false).unwrap();
        assert_eq!(content_type, "application/json");
        let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(parsed["key"], "value");
    }

    #[test]
    fn test_encode_response_msgpack() {
        let data = serde_json::json!({"key": "value"});
        let (content_type, bytes) = encode_response(&data, true).unwrap();
        assert_eq!(content_type, "application/msgpack");
        let parsed: serde_json::Value = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(parsed["key"], "value");
    }

    #[test]
    fn test_payload_offload_threshold() {
        assert_eq!(PAYLOAD_OFFLOAD_THRESHOLD, 1_048_576);
    }

    // --- Fast-path request_id extraction tests ---

    #[test]
    fn test_extract_request_id_fast_array_format() {
        let result = WorkResult {
            work_item_id: "abc-123-def.0".to_string(),
            request_id: "abc-123-def".to_string(),
            item_index: 0,
            success: true,
            result_msgpack: vec![1, 2, 3],
            error: None,
            error_code: None,
            inference_ms: None,
            queue_ms: None,
            processing_ms: None,
            worker_id: None,
            tokenization_ms: None,
            postprocessing_ms: None,
            payload_fetch_ms: None,
        };
        let encoded = rmp_serde::to_vec(&result).unwrap();
        let extracted = extract_request_id_fast(&encoded);
        assert_eq!(extracted, Some("abc-123-def"));
    }

    #[test]
    fn test_extract_request_id_fast_map_format() {
        let result = WorkResult {
            work_item_id: "map-req-456.1".to_string(),
            request_id: "map-req-456".to_string(),
            item_index: 1,
            success: true,
            result_msgpack: vec![10, 20],
            error: None,
            error_code: None,
            inference_ms: None,
            queue_ms: None,
            processing_ms: None,
            worker_id: None,
            tokenization_ms: None,
            postprocessing_ms: None,
            payload_fetch_ms: None,
        };
        let encoded = rmp_serde::to_vec_named(&result).unwrap();
        let extracted = extract_request_id_fast(&encoded);
        // Map format: fast-path scans for "request_id" key and returns its value
        assert_eq!(extracted, Some("map-req-456"));
    }

    #[test]
    fn test_extract_request_id_fast_empty_payload() {
        assert_eq!(extract_request_id_fast(&[]), None);
    }

    #[test]
    fn test_extract_request_id_fast_invalid_payload() {
        // 0xff is a negative fixint (-1), not an array/map — returns None
        assert_eq!(extract_request_id_fast(&[0xff]), None);
        // A single null marker (no string element)
        assert_eq!(extract_request_id_fast(&[0xc0]), None);
        // Truncated data
        assert_eq!(extract_request_id_fast(&[0x91]), None);
    }

    #[test]
    fn test_extract_request_id_fast_uuid() {
        let uuid_str = "550e8400-e29b-41d4-a716-446655440000";
        let result = WorkResult {
            work_item_id: format!("{}.0", uuid_str),
            request_id: uuid_str.to_string(),
            item_index: 0,
            success: true,
            result_msgpack: vec![],
            error: None,
            error_code: None,
            inference_ms: None,
            queue_ms: None,
            processing_ms: None,
            worker_id: None,
            tokenization_ms: None,
            postprocessing_ms: None,
            payload_fetch_ms: None,
        };
        let encoded = rmp_serde::to_vec(&result).unwrap();
        let extracted = extract_request_id_fast(&encoded);
        assert_eq!(extracted, Some(uuid_str));
    }

    #[test]
    fn test_extract_request_id_fast_map_not_first_key() {
        // Build a map where "request_id" is not the first key
        // Use rmp_serde named format on a struct where request_id comes after other fields
        // Since WorkResult has request_id first, we manually build a map
        use rmp::encode::{write_bin, write_map_len, write_str, write_u32};

        let mut buf = Vec::new();
        write_map_len(&mut buf, 4).unwrap();
        // First key: "status"
        write_str(&mut buf, "status").unwrap();
        write_u32(&mut buf, 200).unwrap();
        // Second key: "item_index"
        write_str(&mut buf, "item_index").unwrap();
        write_u32(&mut buf, 0).unwrap();
        // Third key: "request_id"
        write_str(&mut buf, "request_id").unwrap();
        write_str(&mut buf, "found-me").unwrap();
        // Fourth key: "payload"
        write_str(&mut buf, "payload").unwrap();
        write_bin(&mut buf, &[1, 2, 3]).unwrap();

        let extracted = extract_request_id_fast(&buf);
        assert_eq!(extracted, Some("found-me"));
    }

    #[test]
    fn test_skip_msgpack_value_integers() {
        // Positive fixint (0x00..0x7f): single byte
        let data = [0x05, 0xAA]; // fixint 5, then 0xAA
        let rest = skip_msgpack_value(&data).unwrap();
        assert_eq!(rest, &[0xAA]);

        // Negative fixint (0xe0..0xff): single byte
        let data = [0xe0, 0xBB]; // fixint -32, then 0xBB
        let rest = skip_msgpack_value(&data).unwrap();
        assert_eq!(rest, &[0xBB]);

        // u16: marker + 2 bytes
        let data = [0xcd, 0x01, 0x00, 0xCC]; // u16(256), then 0xCC
        let rest = skip_msgpack_value(&data).unwrap();
        assert_eq!(rest, &[0xCC]);
    }

    #[test]
    fn test_skip_msgpack_value_strings() {
        // fixstr "hi" (length 2)
        let data = [0xa2, b'h', b'i', 0xFF];
        let rest = skip_msgpack_value(&data).unwrap();
        assert_eq!(rest, &[0xFF]);
    }

    #[test]
    fn test_skip_msgpack_value_nil_and_bools() {
        // nil
        let data = [0xc0, 0x01];
        let rest = skip_msgpack_value(&data).unwrap();
        assert_eq!(rest, &[0x01]);

        // true
        let data = [0xc3, 0x02];
        let rest = skip_msgpack_value(&data).unwrap();
        assert_eq!(rest, &[0x02]);

        // false
        let data = [0xc2, 0x03];
        let rest = skip_msgpack_value(&data).unwrap();
        assert_eq!(rest, &[0x03]);
    }

    // ── GenerateInput wire-shape regression ──────────────────────

    /// Original prompt-only wire shape: a flat ``{prompt, max_new_tokens, ...}`` map
    /// must still decode into ``GenerateParams { input: Prompt }``. That
    /// guarantees in-flight prompt-only work items remain readable after
    /// the chat-completions surface deploys the enum.
    #[test]
    fn test_generate_params_decodes_slice02_prompt_shape() {
        let wire = rmpv::Value::Map(vec![
            (
                rmpv::Value::from("prompt"),
                rmpv::Value::from("hello world"),
            ),
            (
                rmpv::Value::from("max_new_tokens"),
                rmpv::Value::Integer(32u64.into()),
            ),
            (rmpv::Value::from("temperature"), rmpv::Value::F64(0.7)),
        ]);
        let bytes = rmp_serde::to_vec_named(&wire).unwrap();
        let decoded: GenerateParams = rmp_serde::from_slice(&bytes).unwrap();
        match decoded.input {
            GenerateInput::Prompt { prompt } => assert_eq!(prompt, "hello world"),
            GenerateInput::Messages { .. } => panic!("expected Prompt variant"),
        }
        assert_eq!(decoded.max_new_tokens, 32);
        assert_eq!(decoded.temperature, Some(0.7));
    }

    #[test]
    fn test_generate_params_decodes_messages_shape() {
        let wire = rmpv::Value::Map(vec![
            (
                rmpv::Value::from("messages"),
                rmpv::Value::Array(vec![rmpv::Value::Map(vec![
                    (rmpv::Value::from("role"), rmpv::Value::from("user")),
                    (rmpv::Value::from("content"), rmpv::Value::from("hi")),
                ])]),
            ),
            (
                rmpv::Value::from("max_new_tokens"),
                rmpv::Value::Integer(8u64.into()),
            ),
        ]);
        let bytes = rmp_serde::to_vec_named(&wire).unwrap();
        let decoded: GenerateParams = rmp_serde::from_slice(&bytes).unwrap();
        match decoded.input {
            GenerateInput::Messages { messages } => {
                assert_eq!(messages.len(), 1);
                assert_eq!(messages[0].role, "user");
                assert_eq!(messages[0].content, "hi");
            }
            GenerateInput::Prompt { .. } => panic!("expected Messages variant"),
        }
        assert_eq!(decoded.max_new_tokens, 8);
    }

    /// CRITICAL transport invariant: a generate work item carrying images
    /// must decode as ``serde_json::Value`` — that's the type the sidecar's
    /// ``WorkItem.generate`` uses, and ``serde_json::Value`` CANNOT hold a
    /// msgpack ``bin`` (rmp_serde errors "invalid type: byte array"). Image
    /// bytes therefore travel as a base64 *string* in ``ChatImage.data``.
    /// This test pins that contract: had ``ChatImage.data`` been
    /// ``#[serde(with = "serde_bytes")]`` (bin), the serde_json decode below
    /// would fail and the feature would break through the real
    /// NATS→sidecar→worker path.
    #[test]
    fn test_chat_message_images_survive_sidecar_serde_json_value() {
        let msg = ChatMessage {
            role: "user".to_string(),
            content: "what is this?".to_string(),
            tool_calls: None,
            tool_call_id: None,
            images: Some(vec![ChatImage {
                data: "aGVsbG8=".to_string(), // base64 of b"hello"
                format: Some("png".to_string()),
            }]),
            content_parts: None,
        };
        let mp = rmp_serde::to_vec_named(&msg).unwrap();
        // (1) sidecar step: decode as serde_json::Value — MUST succeed.
        let as_value: serde_json::Value = rmp_serde::from_slice(&mp)
            .expect("ChatMessage with images must decode as serde_json::Value (no msgpack bin)");
        assert_eq!(as_value["images"][0]["data"], "aGVsbG8=");
        // (2) sidecar re-encode → worker typed decode: base64 survives.
        let mp2 = rmp_serde::to_vec_named(&as_value).unwrap();
        let back: ChatMessage = rmp_serde::from_slice(&mp2).unwrap();
        let imgs = back.images.expect("images survive round-trip");
        assert_eq!(imgs[0].data, "aGVsbG8=");
        assert_eq!(imgs[0].format.as_deref(), Some("png"));
    }

    /// ``content_parts`` (the #1294 interleaving layout) must survive the same
    /// NATS→sidecar(``serde_json::Value``)→worker round-trip, preserving the
    /// text↔image ORDER and the internally-tagged shape the worker parses
    /// (``{"type":"text","text":…}`` / ``{"type":"image"}``).
    #[test]
    fn test_content_parts_ordering_survives_sidecar_round_trip() {
        let msg = ChatMessage {
            role: "user".to_string(),
            content: "Page 1:Page 2:which has a cat?".to_string(),
            tool_calls: None,
            tool_call_id: None,
            images: Some(vec![
                ChatImage {
                    data: "aW1nQQ==".to_string(),
                    format: Some("png".to_string()),
                },
                ChatImage {
                    data: "aW1nQg==".to_string(),
                    format: Some("png".to_string()),
                },
            ]),
            content_parts: Some(vec![
                ContentPart::Text {
                    text: "Page 1:".to_string(),
                },
                ContentPart::Image,
                ContentPart::Text {
                    text: "Page 2:".to_string(),
                },
                ContentPart::Image,
                ContentPart::Text {
                    text: "which has a cat?".to_string(),
                },
            ]),
        };
        let mp = rmp_serde::to_vec_named(&msg).unwrap();
        // (1) sidecar step: decode as serde_json::Value — MUST succeed.
        let as_value: serde_json::Value = rmp_serde::from_slice(&mp).expect(
            "ChatMessage with content_parts must decode as serde_json::Value (no msgpack bin)",
        );
        let parts = as_value["content_parts"]
            .as_array()
            .expect("content_parts is an array");
        assert_eq!(parts.len(), 5);
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[0]["text"], "Page 1:");
        assert_eq!(parts[1]["type"], "image");
        assert_eq!(parts[3]["type"], "image");
        assert_eq!(parts[4]["text"], "which has a cat?");
        // (2) sidecar re-encode → worker typed decode: order survives.
        let mp2 = rmp_serde::to_vec_named(&as_value).unwrap();
        let back: ChatMessage = rmp_serde::from_slice(&mp2).unwrap();
        let back_parts = back
            .content_parts
            .expect("content_parts survive round-trip");
        assert!(matches!(&back_parts[0], ContentPart::Text { text } if text == "Page 1:"));
        assert!(matches!(back_parts[1], ContentPart::Image));
        assert!(matches!(back_parts[3], ContentPart::Image));
        assert!(matches!(&back_parts[4], ContentPart::Text { text } if text == "which has a cat?"));
    }

    /// A round-trip through msgpack must preserve the shape — verifies
    /// that the ``flatten`` / ``untagged`` combination encodes the input
    /// arm back into the flat wire shape rather than nesting it under
    /// an ``input:`` key.
    #[test]
    fn test_generate_params_prompt_round_trips_flat() {
        let params = GenerateParams {
            input: GenerateInput::Prompt {
                prompt: "hi".to_string(),
            },
            max_new_tokens: 16,
            ..Default::default()
        };
        let bytes = rmp_serde::to_vec_named(&params).unwrap();
        // Decode back as raw value to assert the flat key set.
        let value: rmpv::Value = rmp_serde::from_slice(&bytes).unwrap();
        let map = match &value {
            rmpv::Value::Map(m) => m,
            _ => panic!("expected map"),
        };
        let keys: Vec<&str> = map
            .iter()
            .filter_map(|(k, _)| match k {
                rmpv::Value::String(s) => s.as_str(),
                _ => None,
            })
            .collect();
        assert!(keys.contains(&"prompt"), "missing prompt key in {keys:?}");
        assert!(
            keys.contains(&"max_new_tokens"),
            "missing max_new_tokens in {keys:?}"
        );
        assert!(!keys.contains(&"input"), "input wrapper leaked: {keys:?}");
        assert!(!keys.contains(&"messages"), "messages leaked: {keys:?}");
    }

    /// Routing-affinity fields appear on the wire when set; otherwise they
    /// are omitted entirely (``skip_serializing_if = "Option::is_none"``)
    /// so a prompt-only worker decoding the bytes still sees the same key
    /// set it expects.
    #[test]
    fn test_work_item_omits_inert_routing_fields_when_none() {
        let item = WorkItem {
            work_item_id: "req-x.0".to_string(),
            request_id: "req-x".to_string(),
            item_index: 0,
            total_items: 1,
            operation: "generate".to_string(),
            model_id: "m".to_string(),
            profile_id: "default".to_string(),
            engine: "pytorch".to_string(),
            pool_name: "p".to_string(),
            admission_pool: String::new(),
            machine_profile: "g".to_string(),
            item: None,
            payload_ref: None,
            output_types: None,
            instruction: None,
            is_query: false,
            options: None,
            query_item: None,
            query_payload_ref: None,
            score_items: None,
            labels: None,
            output_schema: None,
            generate: None,
            routing_key: None,
            prompt_cache_key: None,
            bundle_config_hash: String::new(),
            router_id: String::new(),
            reply_subject: "_INBOX.r.req-x".to_string(),
            timestamp: 1.0,
            traceparent: None,
            tracestate: None,
        };
        let bytes = rmp_serde::to_vec_named(&item).unwrap();
        let value: rmpv::Value = rmp_serde::from_slice(&bytes).unwrap();
        let map = match &value {
            rmpv::Value::Map(m) => m,
            _ => panic!("expected map"),
        };
        let keys: Vec<&str> = map
            .iter()
            .filter_map(|(k, _)| match k {
                rmpv::Value::String(s) => s.as_str(),
                _ => None,
            })
            .collect();
        assert!(
            !keys.contains(&"routing_key"),
            "routing_key leaked when None"
        );
        assert!(
            !keys.contains(&"prompt_cache_key"),
            "prompt_cache_key leaked when None"
        );
    }

    // ── GrammarSpec wire shape ────────────────────────────────────────

    /// Pin the JSON wire shape for ``GrammarSpec::JsonSchema`` against
    /// the Python ``sie_server.types.grammar.GrammarSpec`` dataclass.
    /// Worker-side deserialization reads ``{kind, value, label?,
    /// strict?}`` — any change here is a wire-break.
    #[test]
    fn test_grammar_spec_json_schema_round_trip() {
        let spec = GrammarSpec::JsonSchema {
            value: serde_json::json!({"type": "object", "properties": {"x": {"type": "number"}}}),
            label: Some("math_response".to_string()),
            strict: Some(true),
        };
        let encoded = serde_json::to_value(&spec).expect("serialize");
        // Wire shape check — these field names are Python-readable.
        assert_eq!(encoded["kind"], "json_schema");
        assert_eq!(encoded["value"]["type"], "object");
        assert_eq!(encoded["label"], "math_response");
        assert_eq!(encoded["strict"], true);
        let decoded: GrammarSpec = serde_json::from_value(encoded).expect("round-trip deserialize");
        assert_eq!(decoded, spec);
    }

    #[test]
    fn test_grammar_spec_regex_round_trip() {
        let spec = GrammarSpec::Regex {
            value: r"[A-Z]{3}-\d{4}".to_string(),
            label: None,
            strict: None,
        };
        let encoded = serde_json::to_value(&spec).expect("serialize");
        assert_eq!(encoded["kind"], "regex");
        assert_eq!(encoded["value"], r"[A-Z]{3}-\d{4}");
        // None fields skip-serialise so the worker doesn't see explicit
        // nulls.
        let obj = encoded.as_object().expect("object");
        assert!(!obj.contains_key("label"));
        assert!(!obj.contains_key("strict"));
        let decoded: GrammarSpec = serde_json::from_value(encoded).expect("round-trip deserialize");
        assert_eq!(decoded, spec);
    }

    /// :class:`GenerateParams` carries the grammar through the work
    /// envelope; absence must serialise as field-omitted (not ``null``)
    /// so a prompt-only worker decoding a grammar-bearing work item does not
    /// trip over an unexpected key.
    #[test]
    fn test_generate_params_omits_absent_grammar() {
        let params = GenerateParams {
            input: GenerateInput::Prompt {
                prompt: "Hi".to_string(),
            },
            max_new_tokens: 8,
            ..Default::default()
        };
        let v = serde_json::to_value(&params).expect("serialize");
        let obj = v.as_object().expect("object");
        assert!(
            !obj.contains_key("grammar"),
            "grammar must skip-serialise when None: {v}"
        );
    }

    #[test]
    fn test_generate_params_carries_grammar_when_present() {
        let params = GenerateParams {
            input: GenerateInput::Prompt {
                prompt: "Hi".to_string(),
            },
            max_new_tokens: 8,
            grammar: Some(GrammarSpec::Regex {
                value: r"\d+".to_string(),
                label: None,
                strict: None,
            }),
            ..Default::default()
        };
        let v = serde_json::to_value(&params).expect("serialize");
        assert_eq!(v["grammar"]["kind"], "regex");
        assert_eq!(v["grammar"]["value"], r"\d+");
    }

    // ── M5: W3C Trace Context envelope round-trip ────────────────────

    /// When the gateway has captured a `traceparent` from the inbound
    /// request, it must land on the work envelope verbatim so the
    /// worker can extract it and continue the trace. The two fields
    /// are paired in the wire shape so a single round-trip exercises
    /// both.
    #[test]
    fn test_work_item_carries_traceparent_when_set() {
        let tp = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";
        let ts = "vendor=value";
        let item = WorkItem {
            work_item_id: "req-tp.0".to_string(),
            request_id: "req-tp".to_string(),
            item_index: 0,
            total_items: 1,
            operation: "generate".to_string(),
            model_id: "m".to_string(),
            profile_id: "default".to_string(),
            engine: "pytorch".to_string(),
            pool_name: "p".to_string(),
            admission_pool: String::new(),
            machine_profile: "g".to_string(),
            item: None,
            payload_ref: None,
            output_types: None,
            instruction: None,
            is_query: false,
            options: None,
            query_item: None,
            query_payload_ref: None,
            score_items: None,
            labels: None,
            output_schema: None,
            generate: None,
            routing_key: None,
            prompt_cache_key: None,
            bundle_config_hash: String::new(),
            router_id: String::new(),
            reply_subject: "_INBOX.r.req-tp".to_string(),
            timestamp: 1.0,
            traceparent: Some(tp.to_string()),
            tracestate: Some(ts.to_string()),
        };
        let bytes = rmp_serde::to_vec_named(&item).unwrap();
        let decoded: WorkItem = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(decoded.traceparent.as_deref(), Some(tp));
        assert_eq!(decoded.tracestate.as_deref(), Some(ts));
    }

    /// Backward-compat: when both trace fields are absent, the
    /// msgpack must omit them entirely (not encode `null`), so a
    /// pre-M5 worker reading the bytes sees its expected key set.
    #[test]
    fn test_work_item_omits_trace_fields_when_none() {
        let item = WorkItem {
            work_item_id: "req-tp2.0".to_string(),
            request_id: "req-tp2".to_string(),
            item_index: 0,
            total_items: 1,
            operation: "generate".to_string(),
            model_id: "m".to_string(),
            profile_id: "default".to_string(),
            engine: "pytorch".to_string(),
            pool_name: "p".to_string(),
            admission_pool: String::new(),
            machine_profile: "g".to_string(),
            item: None,
            payload_ref: None,
            output_types: None,
            instruction: None,
            is_query: false,
            options: None,
            query_item: None,
            query_payload_ref: None,
            score_items: None,
            labels: None,
            output_schema: None,
            generate: None,
            routing_key: None,
            prompt_cache_key: None,
            bundle_config_hash: String::new(),
            router_id: String::new(),
            reply_subject: "_INBOX.r.req-tp2".to_string(),
            timestamp: 1.0,
            traceparent: None,
            tracestate: None,
        };
        let bytes = rmp_serde::to_vec_named(&item).unwrap();
        let value: rmpv::Value = rmp_serde::from_slice(&bytes).unwrap();
        let map = match &value {
            rmpv::Value::Map(m) => m,
            _ => panic!("expected map"),
        };
        let keys: Vec<&str> = map
            .iter()
            .filter_map(|(k, _)| match k {
                rmpv::Value::String(s) => s.as_str(),
                _ => None,
            })
            .collect();
        assert!(
            !keys.contains(&"traceparent"),
            "traceparent leaked when None: {keys:?}"
        );
        assert!(
            !keys.contains(&"tracestate"),
            "tracestate leaked when None: {keys:?}"
        );
    }

    /// End-to-end gateway round-trip: an active span on the gateway
    /// side must yield a `traceparent` on the envelope. Uses the same
    /// `inject_current_context` helper the publisher calls in
    /// production, so this locks the integration point.
    #[test]
    fn test_inject_current_context_with_active_span_populates_envelope() {
        use opentelemetry::trace::{TraceContextExt, Tracer, TracerProvider as _};
        use opentelemetry::Context;
        use opentelemetry_sdk::propagation::TraceContextPropagator;
        use opentelemetry_sdk::trace::TracerProvider;

        opentelemetry::global::set_text_map_propagator(TraceContextPropagator::new());
        let provider = TracerProvider::builder().build();
        let tracer = provider.tracer("gateway-test");
        let span = tracer.start("gateway.proxy_chat");
        let cx = Context::current().with_span(span);
        let _guard = cx.attach();

        let (tp, _ts) = crate::observability::propagation::inject_current_context();
        let tp = tp.expect("active span must produce a traceparent");
        let parts: Vec<&str> = tp.split('-').collect();
        assert_eq!(parts.len(), 4, "wire shape: version-trace-span-flags");
        assert_eq!(parts[1].len(), 32, "trace_id is 32 hex chars");
        assert_eq!(parts[2].len(), 16, "span_id is 16 hex chars");
    }

    #[test]
    fn test_queue_trace_context_stays_off_for_non_generation() {
        use opentelemetry::trace::{TraceContextExt, Tracer, TracerProvider as _};
        use opentelemetry::Context;
        use opentelemetry_sdk::propagation::TraceContextPropagator;
        use opentelemetry_sdk::trace::TracerProvider;

        opentelemetry::global::set_text_map_propagator(TraceContextPropagator::new());
        let provider = TracerProvider::builder().build();
        let tracer = provider.tracer("gateway-test");
        let span = tracer.start("gateway.proxy_request");
        let cx = Context::current().with_span(span);
        let _guard = cx.attach();

        for endpoint in InferenceEndpoint::NON_GENERATION_QUEUE_LABELS {
            assert!(
                !WorkPublisher::should_propagate_queue_trace(endpoint),
                "{endpoint} must keep the pre-generation queue hot path"
            );
            let (tp, ts) = WorkPublisher::trace_context_for_endpoint(endpoint);
            assert!(tp.is_none(), "{endpoint} unexpectedly injected traceparent");
            assert!(ts.is_none(), "{endpoint} unexpectedly injected tracestate");
        }
        let (tp, ts) = WorkPublisher::trace_context_for_endpoint("unknown");
        assert!(
            tp.is_none(),
            "unknown endpoint unexpectedly injected traceparent"
        );
        assert!(
            ts.is_none(),
            "unknown endpoint unexpectedly injected tracestate"
        );
    }

    #[test]
    fn test_queue_trace_context_is_generation_only() {
        use opentelemetry::trace::{TraceContextExt, Tracer, TracerProvider as _};
        use opentelemetry::Context;
        use opentelemetry_sdk::propagation::TraceContextPropagator;
        use opentelemetry_sdk::trace::TracerProvider;

        opentelemetry::global::set_text_map_propagator(TraceContextPropagator::new());
        let provider = TracerProvider::builder().build();
        let tracer = provider.tracer("gateway-test");
        let span = tracer.start("gateway.proxy_generate");
        let cx = Context::current().with_span(span);
        let _guard = cx.attach();

        assert!(WorkPublisher::should_propagate_queue_trace("generate"));
        let (tp, _ts) = WorkPublisher::trace_context_for_endpoint("generate");
        let tp = tp.expect("generate queue fallback should propagate traceparent");
        let parts: Vec<&str> = tp.split('-').collect();
        assert_eq!(parts.len(), 4, "wire shape: version-trace-span-flags");
    }

    // ----- H9 — first-chunk-fallback rate limit ---------------------

    /// The bucket admits exactly ``burst`` tokens immediately, then
    /// refuses until the rate refills enough for the next whole token.
    /// Refilling is driven by monotonic time so the test sleeps briefly
    /// to advance the clock; the timing window is generous enough to
    /// avoid flakes on a loaded CI runner.
    #[test]
    fn test_token_bucket_burst_and_refill() {
        let mut bucket = TokenBucket::new(10.0, 3.0); // 10/s, burst 3
                                                      // Burst exhausted in three takes.
        assert!(bucket.try_take());
        assert!(bucket.try_take());
        assert!(bucket.try_take());
        assert!(!bucket.try_take(), "burst exceeded — must refuse");

        // Sleep ~120ms — at 10/s that's >=1 token of refill.
        std::thread::sleep(Duration::from_millis(150));
        assert!(bucket.try_take(), "expected refill to permit one more take");
    }

    /// Regression for the burst cap: the bucket must NOT accumulate
    /// tokens above ``burst`` even when idle for a long stretch. Without
    /// the cap a quiet system would let a single noisy request drain
    /// hours of accrued tokens at once.
    #[test]
    fn test_token_bucket_caps_at_burst() {
        let mut bucket = TokenBucket::new(100.0, 2.0); // very fast refill, tiny burst
        std::thread::sleep(Duration::from_millis(50)); // would refill 5 tokens uncapped
                                                       // Two takes succeed (burst), the third refuses (no overflow stored).
        assert!(bucket.try_take());
        assert!(bucket.try_take());
        assert!(
            !bucket.try_take(),
            "burst cap must be enforced even after a long idle window"
        );
    }

    /// Property: drop a flurry of N > burst tries with zero sleep —
    /// exactly ``burst`` succeed. Mirrors the gateway-side scenario where
    /// a cold-start storm fires more first-chunk timeouts than the rate
    /// permits and we want a deterministic, bounded number of republishes.
    #[test]
    fn test_token_bucket_drops_excess_attempts() {
        let mut bucket = TokenBucket::new(5.0, 4.0); // 5/s, burst 4
        let mut admitted = 0usize;
        for _ in 0..20 {
            if bucket.try_take() {
                admitted += 1;
            }
        }
        assert_eq!(
            admitted, 4,
            "exactly burst-many attempts admit when the test runs faster than the refill rate"
        );
    }

    /// Key-isolation test for the per-(model, pool) bucket map: the
    /// rate limit must NOT cross-talk between distinct key tuples.
    /// Models the gateway-side scenario where one pool is in a fallback
    /// storm and a different pool's healthy traffic must be unaffected.
    /// We exercise the keying logic directly via the same DashMap +
    /// TokenBucket types the production code uses — no NATS / JetStream
    /// dependency, no I/O.
    #[test]
    fn test_fallback_rate_limit_isolates_keys_and_drops_excess() {
        let buckets: DashMap<String, std::sync::Mutex<TokenBucket>> = DashMap::new();
        let rate = FALLBACK_RATE_PER_SEC_DEFAULT;
        let burst = FALLBACK_BURST_DEFAULT;

        fn try_take(
            buckets: &DashMap<String, std::sync::Mutex<TokenBucket>>,
            rate: f64,
            burst: f64,
            model: &str,
            pool: &str,
        ) -> bool {
            let key = format!("{}|{}", model, pool);
            let entry = buckets
                .entry(key)
                .or_insert_with(|| std::sync::Mutex::new(TokenBucket::new(rate, burst)));
            let admitted = entry.value().lock().unwrap().try_take();
            admitted
        }

        // Same key — admit exactly burst, then refuse.
        let mut admitted_a = 0usize;
        for _ in 0..30 {
            if try_take(&buckets, rate, burst, "model-A", "pool-1") {
                admitted_a += 1;
            }
        }
        assert_eq!(admitted_a, burst as usize);

        // Different model on same pool → independent bucket → full burst.
        let mut admitted_b = 0usize;
        for _ in 0..30 {
            if try_take(&buckets, rate, burst, "model-B", "pool-1") {
                admitted_b += 1;
            }
        }
        assert_eq!(admitted_b, burst as usize);

        // Different pool on same model → also independent.
        let mut admitted_c = 0usize;
        for _ in 0..30 {
            if try_take(&buckets, rate, burst, "model-A", "pool-2") {
                admitted_c += 1;
            }
        }
        assert_eq!(admitted_c, burst as usize);
    }
}
