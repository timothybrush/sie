---
name: superlinked-docs
description: >-
  Offload document, image, and structured-output work to the Superlinked
  inference cluster: convert PDF/DOCX/PPTX/XLSX/HTML/scans to clean markdown,
  describe an image (caption + tags), or produce schema/grammar-constrained JSON
  off the cluster — instead of ingesting the file directly, which can reduce
  the tokens billed in many cases for document- and image-heavy work. Use
  whenever the user drops or references a document or image file to read,
  summarize, extract from, describe, or answer questions over, or when they need
  reliable structured (schema-valid) output.
---

# Superlinked document offload

When the user gives you a document (PDF, DOCX, PPTX, XLSX, HTML, or a scan), do **not**
open or attach the file into the conversation — that bills every page as text *and* a
page-image. Convert it to markdown on the Superlinked cluster first, then work from the
markdown.

## docs → markdown

1. Read the source file as **raw bytes** and base64-encode it. Do not view or attach it.
2. Call the `docs_to_markdown` MCP tool with:
   - `document_base64`: the base64-encoded bytes
   - `filename`: the original filename (its extension hints the format)
   - `engine`: leave as `auto` (default) — it runs the standard converter and
     automatically falls back to a vision OCR model for scanned/image-only pages, so
     scanned PDFs need no extra flag. Force `vl-ocr` for scanned or complex-layout
     PDFs you want OCR'd page-by-page, or `docling` to pin the standard converter.
   - `ocr`: leave `false`. It only turns on the standard converter's built-in OCR
     under `engine: docling`; under `auto` the vision-OCR fallback handles scans, so
     `ocr` is ignored there.
3. Write the returned `markdown` to a file named after the source, then read and reason
   over **that** file — never the original. Where to write it depends on the surface:
   - **Cowork** (connected folder): `.superlinked/<original-name>.md` (create `.superlinked/` if absent).
   - **claude.ai / Claude desktop app** (code-execution sandbox): `./<original-name>.md` in the
     working directory; offer to export it (Download, save to Google Drive, or add to a
     Project's knowledge to reuse it across chats).
   - **Claude Code** (local filesystem): alongside the source.
4. From here on, read and reason over the saved `.md` — never the original.

The tool response also includes a `metadata` object with two distinct figures: a live
per-call `markdown_tokens_estimate` (a rough ~4 chars/token count of *this* response — an
estimate, not a billed figure) and `token_reduction`, the committed #1311 benchmark
measurement of markdown vs direct document ingestion. Surface these if the user asks how
much was saved; the exact percentages and the benchmark path are in the metadata.

## Structured output (schema-valid JSON / constrained generation)

Two tools constrain the cluster's output at decode time, so the result is a reliable,
machine-parseable artifact instead of free text. Prefer these over asking a model to
"return JSON" — the constraint is enforced by the serving engine's grammar backend.

### extract → structured record

To pull a structured record out of text or markdown (e.g. the output of
`docs_to_markdown`), call the `extract_structured` MCP tool with:

- `content`: the source text/markdown to extract from
- `output_schema`: a JSON Schema describing the record
- `instruction` (optional): extra guidance on what to pull out
- `model` (optional): generation model override (defaults to the cluster's configured structured-output model)
- `max_output_tokens` (optional): raise the output-token ceiling for large records

You get back `{ "data": <object> }`. The model is constrained to `output_schema` at
decode time **and** the result is validated against it before return — if the serving
profile bypasses the constraint, you get a clear error instead of non-conforming JSON.
Values are grounded in `content` — the model is told not to invent them.

### generate → constrained output

To generate output under a constraint, call the `generate_structured` MCP tool with:

- `prompt`: the instruction to answer
- `response_format`: one of
  - `{"type": "json_schema", "json_schema": {"name": ..., "schema": <schema>, "strict": true}}`
  - `{"type": "json_object"}`
  - `{"type": "regex", "regex": "<pattern>"}`
  - `{"type": "grammar", "grammar": "<ebnf>", "syntax": "ebnf"}` (xgrammar-backed models only)
- `model` (optional): generation model override (defaults to the cluster's configured structured-output model)
- `max_output_tokens` (optional): raise the output-token ceiling for large output

You get back `{ "content": "<string>" }` (a JSON string for the json modes). A
`json_schema` result is validated against the schema before return.

### Schema limits

`output_schema` (and a `json_schema` `response_format`) must be in the Outlines-supported
subset: **no `$ref`** (so no recursion), **no conditionals** (`if`/`then`/`else`), nesting
**depth ≤ 16**. The schema must also stay within the gateway safety caps (≤ 16 384 nodes,
≤ 64 KiB serialized; `regex` ≤ 4 KiB). Out-of-subset or over-cap schemas are rejected up
front with a clear error. The default model is Outlines-backed, so **EBNF/`grammar` is not
available** on it — use `json_schema` or `regex` instead; an EBNF request is rejected with a
clear error.

## describe image

When the user gives you an image (PNG/JPEG) to caption, tag, or answer questions over,
do **not** open or attach the image into the conversation — that bills the image tokens.
Describe it on the Superlinked cluster first, then work from the returned text.

1. Read the image file as **raw bytes** and base64-encode it. Do not view or attach it.
2. Call the `describe_image` MCP tool with:
   - `image_base64`: the base64-encoded bytes
   - `labels` (optional): candidate tags to score against; omit to use the service's
     default label set
   - `detailed` (optional): `true` for a longer Florence-2 `<DETAILED_CAPTION>`
   - `top_k` (optional): how many of the top-scoring tags to return
3. You get back `caption` (a Florence-2 caption) and `tags` (a list of
   `{label, score}` ranked by zero-shot similarity to the image). Reason over those
   instead of the raw image.

The caption is produced by Florence-2; the tags are zero-shot — the cluster embeds the
candidate labels and the image (SigLIP/CLIP) and returns the closest labels. No image
pixels reach the calling model.

## Authentication

`docs_to_markdown`, `describe_image`, `extract_structured`, and `generate_structured` are all
served by the Superlinked MCP connector. Configure the connector with your **connector
secret** — credentials are never pasted into the chat. For this POC the connector secret is
provided by your cluster operator; self-serve issuance arrives with the managed service.
