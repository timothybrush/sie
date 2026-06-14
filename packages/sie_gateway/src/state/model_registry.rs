use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use arc_swap::ArcSwap;
use sha2::{Digest, Sha256};
use tracing::{debug, error, info, warn};

use crate::types::bundle::{engine_adapter_prefixes, BundleInfo, DEFAULT_ENGINE, KNOWN_ENGINES};
use crate::types::model::{
    CanonicalProfile, ModelConfig, ModelEntry, ModelInfoExtras, ProfileConfig,
};

#[derive(Debug)]
pub struct ModelNotFoundError {
    pub model: String,
}

impl std::fmt::Display for ModelNotFoundError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Model not found: {}", self.model)
    }
}

impl std::error::Error for ModelNotFoundError {}

#[derive(Debug)]
pub struct BundleConflictError {
    pub model: String,
    pub bundle: String,
    pub compatible_bundles: Vec<String>,
}

impl std::fmt::Display for BundleConflictError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Bundle '{}' does not support model '{}'. Compatible bundles: {:?}",
            self.bundle, self.model, self.compatible_bundles
        )
    }
}

impl std::error::Error for BundleConflictError {}

#[derive(Debug)]
pub enum ResolveError {
    ModelNotFound(ModelNotFoundError),
    BundleConflict(BundleConflictError),
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResolveError::ModelNotFound(e) => write!(f, "{}", e),
            ResolveError::BundleConflict(e) => write!(f, "{}", e),
        }
    }
}

impl std::error::Error for ResolveError {}

/// Outcome of `add_model_config`: `(created_profiles, skipped_profiles,
/// affected_bundles)`. Surfaces through to the NATS publish path so the
/// caller can log which profiles actually changed vs. were idempotently
/// skipped, and which bundles need to be re-broadcast to workers.
pub type AddModelConfigOutcome = (Vec<String>, Vec<String>, Vec<String>);

const DEFAULT_MODEL_POOL: &str = "default";
const MAX_POOL_NAME_LEN: usize = 128;

#[derive(Debug, Clone, Default)]
struct RegistrySnapshot {
    bundles: HashMap<String, BundleInfo>,
    models: HashMap<String, ModelEntry>,
    model_names_lower: HashMap<String, String>,
    bundle_config_hashes: HashMap<String, String>,
    bundle_pool_config_hashes: HashMap<(String, String), String>,
}

pub struct ModelRegistry {
    bundles_dir: PathBuf,
    models_dir: PathBuf,
    snapshot: ArcSwap<RegistrySnapshot>,
    /// Serializes the read-modify-write path. All mutators
    /// (`add_model_config`, `reload`) take this before `snapshot.load()` and
    /// release it after `snapshot.store(...)` so two concurrent writers
    /// cannot both build derived snapshots off the same base and lose each
    /// other's mutations. Readers still use `snapshot.load()` unlocked and
    /// see a consistent `Arc<RegistrySnapshot>` per access.
    write_lock: Mutex<()>,
}

impl ModelRegistry {
    pub fn new(
        bundles_dir: impl AsRef<Path>,
        models_dir: impl AsRef<Path>,
        auto_load: bool,
    ) -> Self {
        let registry = Self {
            bundles_dir: bundles_dir.as_ref().to_path_buf(),
            models_dir: models_dir.as_ref().to_path_buf(),
            snapshot: ArcSwap::from_pointee(RegistrySnapshot::default()),
            write_lock: Mutex::new(()),
        };
        if auto_load {
            registry.reload();
        }
        registry
    }

    pub fn bundles_dir(&self) -> &Path {
        &self.bundles_dir
    }

    pub fn models_dir(&self) -> &Path {
        &self.models_dir
    }

    fn canonical_model_name(snapshot: &RegistrySnapshot, model: &str) -> Option<String> {
        snapshot
            .model_names_lower
            .get(&model.to_lowercase())
            .cloned()
            .or_else(|| {
                if snapshot.models.contains_key(model) {
                    Some(model.to_string())
                } else {
                    None
                }
            })
    }

    pub fn reload(&self) {
        // Serialize vs add_model_config so a concurrent filesystem reload
        // cannot stomp on a partially-built snapshot.
        let _write = self
            .write_lock
            .lock()
            .expect("ModelRegistry write_lock poisoned");
        let mut new_bundles: HashMap<String, BundleInfo> = HashMap::new();
        let mut new_models: HashMap<String, ModelEntry> = HashMap::new();
        let mut new_model_names_lower: HashMap<String, String> = HashMap::new();

        // Load bundles
        if self.bundles_dir.exists() {
            match std::fs::read_dir(&self.bundles_dir) {
                Ok(entries) => {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if path.extension().and_then(|e| e.to_str()) != Some("yaml") {
                            continue;
                        }
                        match Self::load_bundle_file(&path) {
                            Ok(bundle) => {
                                debug!(
                                    bundle = %bundle.name,
                                    priority = bundle.priority,
                                    adapters = bundle.adapters.len(),
                                    "loaded bundle"
                                );
                                new_bundles.insert(bundle.name.clone(), bundle);
                            }
                            Err(e) => {
                                error!(path = %path.display(), error = %e, "failed to load bundle");
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!(dir = %self.bundles_dir.display(), error = %e, "cannot read bundles directory");
                }
            }
        } else {
            warn!(dir = %self.bundles_dir.display(), "bundles directory not found");
        }

        // Load models
        if self.models_dir.exists() {
            match std::fs::read_dir(&self.models_dir) {
                Ok(entries) => {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if !path.is_file() {
                            continue;
                        }
                        if path.extension().and_then(|e| e.to_str()) != Some("yaml") {
                            continue;
                        }
                        match Self::load_model_file(&path) {
                            Ok(model_entries) => {
                                for model_entry in model_entries {
                                    debug!(
                                        model = %model_entry.name,
                                        adapters = ?model_entry.adapter_modules,
                                        "discovered model"
                                    );
                                    new_model_names_lower.insert(
                                        model_entry.name.to_lowercase(),
                                        model_entry.name.clone(),
                                    );
                                    new_models.insert(model_entry.name.clone(), model_entry);
                                }
                            }
                            Err(e) => {
                                error!(path = %path.display(), error = %e, "failed to load model config");
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!(dir = %self.models_dir.display(), error = %e, "cannot read models directory");
                }
            }
        } else {
            warn!(dir = %self.models_dir.display(), "models directory not found");
        }

        // Compute model-to-bundle mappings
        for model_entry in new_models.values_mut() {
            Self::assign_bundles(model_entry, &new_bundles);
        }

        // Pre-compute bundle config hashes (expensive: sort + JSON + SHA-256).
        // Cached here so route/status lookups are simple HashMap reads.
        let bundle_config_hashes = Self::rebuild_bundle_config_hashes(&new_bundles, &new_models);
        let bundle_pool_config_hashes =
            Self::rebuild_bundle_pool_config_hashes(&new_bundles, &new_models);

        info!(
            bundles = new_bundles.len(),
            models = new_models.len(),
            "model registry loaded"
        );

        let snap = RegistrySnapshot {
            bundles: new_bundles,
            models: new_models,
            model_names_lower: new_model_names_lower,
            bundle_config_hashes,
            bundle_pool_config_hashes,
        };
        self.snapshot.store(Arc::new(snap));
    }

    fn load_bundle_file(path: &Path) -> Result<BundleInfo, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let data: serde_yaml::Value = serde_yaml::from_str(&content)?;

        let name = data
            .get("name")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| {
                path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string()
            });

        let priority = data.get("priority").and_then(|v| v.as_i64()).unwrap_or(100) as i32;

        let adapters: Vec<String> = data
            .get("adapters")
            .and_then(|v| v.as_sequence())
            .map(|seq| {
                seq.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        // Engine: defaults to "pytorch" for back-compat with bundle
        // YAMLs written before this field existed. Unknown values
        // are a hard error — silently coercing to the default would
        // let a typo'd ``engine: pytroch`` mis-route traffic and
        // push diagnosis to the symptom side. Mirrors the
        // equivalent check in ``sie_config/model_registry.py::reload``.
        let engine = data
            .get("engine")
            .and_then(|v| v.as_str())
            .unwrap_or(DEFAULT_ENGINE)
            .to_string();
        if !KNOWN_ENGINES.contains(&engine.as_str()) {
            return Err(format!(
                "bundle {:?} declares engine={:?} which is not in KNOWN_ENGINES={:?}; \
                 update the bundle YAML or add the engine to types::bundle::KNOWN_ENGINES",
                name, engine, KNOWN_ENGINES,
            )
            .into());
        }

        // Adapter-namespace consistency check. The matcher
        // (``resolve_bundle`` and ``compute_bundle_config_hash``)
        // intersects ``bundle.adapters`` with each model's
        // ``adapter_path`` modules — so a bundle that accidentally
        // lists adapters outside this engine's namespace would
        // produce ``UnsupportedModel`` IPC NAKs at runtime.
        // Catching this at config-load time turns it into a
        // deploy-time signal.
        let expected_prefixes = engine_adapter_prefixes(&engine);
        if !expected_prefixes.is_empty() {
            let bad: Vec<&String> = adapters
                .iter()
                .filter(|a| !expected_prefixes.iter().any(|p| a.starts_with(p)))
                .collect();
            if !bad.is_empty() {
                warn!(
                    bundle = %name,
                    engine = %engine,
                    expected = ?expected_prefixes,
                    bad_adapters = ?bad,
                    "bundle lists adapter(s) outside the expected namespace(s); \
                     these will not be servable by a worker speaking this engine"
                );
            }
        }

        Ok(BundleInfo {
            name,
            priority,
            adapters,
            engine,
        })
    }

    fn load_model_file(path: &Path) -> Result<Vec<ModelEntry>, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: ModelConfig = serde_yaml::from_str(&content)?;

        Self::expand_model_config_into_profile_variants(&config).map_err(|message| {
            Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                message,
            )) as Box<dyn std::error::Error>
        })
    }

    /// Profile names show up in the registry as `format!("{base}:{profile}")`.
    /// A name containing `/`, `:`, whitespace, or non-printable bytes would
    /// either shadow a legitimate base model id or produce an unparseable
    /// model spec when concatenated. Reject at load so misconfigured YAML
    /// fails fast instead of producing a phantom registry entry that
    /// routing then silently bypasses.
    fn validate_profile_name(name: &str) -> Result<(), String> {
        if name.is_empty() {
            return Err("profile name must not be empty".to_string());
        }
        if !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
        {
            return Err(format!(
                "profile name '{name}' must match [A-Za-z0-9._-]+ \
                 (no '/', ':', whitespace, or other punctuation)"
            ));
        }
        Ok(())
    }

    fn expand_model_config_into_profile_variants(
        config: &ModelConfig,
    ) -> Result<Vec<ModelEntry>, String> {
        let base_entry = Self::model_entry_from_config(config)?;
        let mut entries = vec![base_entry.clone()];
        let mut profile_names: Vec<String> = base_entry.profile_names.iter().cloned().collect();
        profile_names.sort();
        for profile_name in profile_names {
            if profile_name == "default" {
                continue;
            }
            Self::validate_profile_name(&profile_name)?;
            let profile_config =
                Self::resolved_profile_from_profiles(&config.profiles, &profile_name)?;
            let mut variant_profile_names = HashSet::new();
            variant_profile_names.insert("default".to_string());
            let mut variant_profile_configs = HashMap::new();
            variant_profile_configs.insert("default".to_string(), profile_config);
            let mut variant_adapters = HashSet::new();
            if let Some(adapter_path) =
                Self::effective_adapter_path_from_profiles(&config.profiles, &profile_name)
            {
                if let Some(module) = Self::adapter_module_from_path(adapter_path) {
                    variant_adapters.insert(module);
                }
            }
            entries.push(ModelEntry {
                name: format!("{}:{}", base_entry.name, profile_name),
                pool: base_entry.pool.clone(),
                bundles: Vec::new(),
                adapter_modules: variant_adapters,
                profile_names: variant_profile_names,
                profile_configs: variant_profile_configs,
                info_extras: Self::narrow_info_extras_to_profile(
                    &base_entry.info_extras,
                    &profile_name,
                ),
            });
        }

        Ok(entries)
    }

    /// Re-scope an ``info_extras`` (built for the base entry, which holds
    /// the union of capabilities across profiles) so a variant entry
    /// ``{base}:{profile_name}`` only advertises and validates against the
    /// adapters configured for *its* profile. Without this narrowing,
    /// the variant inherits the union and the lora_adapter gate accepts
    /// adapters that are only configured on a sibling profile (M10 bug).
    /// All other capability fields stay the same.
    fn narrow_info_extras_to_profile(
        base: &ModelInfoExtras,
        profile_name: &str,
    ) -> ModelInfoExtras {
        let mut narrowed = base.clone();
        if let Some(map) = base.profile_lora_adapters.as_ref() {
            let scoped = map.get(profile_name).cloned().unwrap_or_default();
            if scoped.is_empty() {
                // Profile has no adapters declared — drop both fields so
                // the gate rejects any ``lora_adapter`` value as unknown.
                narrowed.lora_adapters = None;
                narrowed.profile_lora_adapters = None;
            } else {
                narrowed.lora_adapters = Some(scoped.clone());
                let mut single = HashMap::new();
                // Variants are keyed under ``"default"`` in their
                // ``profile_configs`` (the resolved profile becomes the
                // variant's default); mirror that here so the validation
                // gate's profile lookup (``"default"`` for variants)
                // matches the narrowed map.
                single.insert("default".to_string(), scoped);
                narrowed.profile_lora_adapters = Some(single);
            }
        }
        narrowed
    }

    fn model_entry_from_config(config: &ModelConfig) -> Result<ModelEntry, String> {
        let info_extras = crate::types::model::ModelInfoExtras::from_model_config(config);
        let model_name = config.name.clone();
        let pool = Self::normalize_model_pool(config.pool.as_deref())?;
        let mut adapter_modules: HashSet<String> = HashSet::new();
        let mut profile_names: HashSet<String> = HashSet::new();
        let mut profile_configs: HashMap<String, CanonicalProfile> = HashMap::new();

        for profile_name in config.profiles.keys() {
            profile_names.insert(profile_name.clone());
            if let Some(adapter_path) =
                Self::effective_adapter_path_from_profiles(&config.profiles, profile_name)
            {
                if let Some(module) = Self::adapter_module_from_path(adapter_path) {
                    adapter_modules.insert(module);
                }
            }
            profile_configs.insert(
                profile_name.clone(),
                Self::resolved_profile_from_profiles(&config.profiles, profile_name)?,
            );
        }

        Ok(ModelEntry {
            name: model_name,
            pool,
            bundles: Vec::new(),
            adapter_modules,
            profile_names,
            profile_configs,
            info_extras,
        })
    }

    fn normalize_model_pool(pool: Option<&str>) -> Result<Option<String>, String> {
        let Some(raw_pool) = pool else {
            return Ok(None);
        };
        let pool = raw_pool.trim().to_lowercase();
        if pool.is_empty() || pool == DEFAULT_MODEL_POOL {
            return Ok(None);
        }
        if pool.len() > MAX_POOL_NAME_LEN
            || pool == "_default"
            || !pool
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        {
            return Err(
                "Field 'pool' must match [A-Za-z0-9_-]{1,128} and must not be _default".to_string(),
            );
        }
        Ok(Some(pool))
    }

    fn entry_pool_name(entry: &ModelEntry) -> &str {
        entry.pool.as_deref().unwrap_or(DEFAULT_MODEL_POOL)
    }

    fn expand_model_entry_into_profile_variants(
        base_entry: &ModelEntry,
    ) -> Result<Vec<ModelEntry>, String> {
        let mut entries = vec![base_entry.clone()];
        let mut profile_names: Vec<String> = base_entry.profile_names.iter().cloned().collect();
        profile_names.sort();
        for profile_name in profile_names {
            if profile_name == "default" {
                continue;
            }
            // Enforce the same profile-name allowlist as the file-load
            // expansion path (`expand_model_config_into_profile_variants`).
            // This is the LIVE delta path
            // (`add_model_config_inner` → here): without this check a
            // control-plane delta carrying a profile named e.g.
            // `bad/name` or `a:b` would mint a phantom registry entry
            // (`{base}:{bad/name}`) that produces an illegal model spec /
            // shadows a base id, which routing then silently bypasses.
            Self::validate_profile_name(&profile_name)?;
            let Some(profile_config) = Self::resolved_profile_from_entry(base_entry, &profile_name)
            else {
                return Err(Self::profile_resolution_error(&profile_name));
            };
            let mut variant_profile_names = HashSet::new();
            variant_profile_names.insert("default".to_string());
            let mut variant_profile_configs = HashMap::new();
            variant_profile_configs.insert("default".to_string(), profile_config);
            let mut variant_adapters = HashSet::new();
            if let Some(adapter_path) =
                Self::effective_adapter_path_from_entry(base_entry, &profile_name)
            {
                if let Some(module) = Self::adapter_module_from_path(adapter_path) {
                    variant_adapters.insert(module);
                }
            }
            entries.push(ModelEntry {
                name: format!("{}:{}", base_entry.name, profile_name),
                pool: base_entry.pool.clone(),
                bundles: Vec::new(),
                adapter_modules: variant_adapters,
                profile_names: variant_profile_names,
                profile_configs: variant_profile_configs,
                info_extras: Self::narrow_info_extras_to_profile(
                    &base_entry.info_extras,
                    &profile_name,
                ),
            });
        }

        Ok(entries)
    }

    fn adapter_module_from_path(adapter_path: &str) -> Option<String> {
        if adapter_path.is_empty() {
            None
        } else {
            Some(
                adapter_path
                    .split(':')
                    .next()
                    .unwrap_or(adapter_path)
                    .to_string(),
            )
        }
    }

    fn empty_profile_config() -> crate::types::model::ProfileConfig {
        crate::types::model::ProfileConfig {
            adapter_path: None,
            max_batch_tokens: None,
            compute_precision: None,
            adapter_options: None,
            extends: None,
        }
    }

    fn resolved_profile_from_profiles(
        profiles: &HashMap<String, crate::types::model::ProfileConfig>,
        profile_name: &str,
    ) -> Result<CanonicalProfile, String> {
        fn resolve(
            profiles: &HashMap<String, crate::types::model::ProfileConfig>,
            profile_name: &str,
            seen: &mut HashSet<String>,
        ) -> Option<crate::types::model::ProfileConfig> {
            if !seen.insert(profile_name.to_string()) {
                return None;
            }
            let profile = profiles.get(profile_name)?;
            let mut resolved = if let Some(parent_name) = profile.extends.as_deref() {
                resolve(profiles, parent_name, seen)?
            } else {
                ModelRegistry::empty_profile_config()
            };

            if profile.adapter_path.is_some() {
                resolved.adapter_path = profile.adapter_path.clone();
            }
            if profile.max_batch_tokens.is_some() {
                resolved.max_batch_tokens = profile.max_batch_tokens;
            }
            if profile.compute_precision.is_some() {
                resolved.compute_precision = profile.compute_precision.clone();
            }
            if profile.adapter_options.is_some() {
                resolved.adapter_options = profile.adapter_options.clone();
            }
            resolved.extends = None;
            Some(resolved)
        }

        resolve(profiles, profile_name, &mut HashSet::new())
            .map(|profile| CanonicalProfile::from_profile(&profile))
            .ok_or_else(|| Self::profile_resolution_error(profile_name))
    }

    fn profile_resolution_error(profile_name: &str) -> String {
        format!("Profile '{profile_name}' has a missing parent or an inheritance cycle")
    }

    /// Build a ``ProfileConfig`` map that overlays incoming profiles on top
    /// of the canonical profiles already stored for an existing model.
    /// Used by the update path so a delta config whose new profile
    /// ``extends`` a profile that lives only in
    /// ``existing.profile_configs`` still resolves cleanly. Incoming
    /// profiles win on collisions because the caller's whole point is to
    /// re-evaluate them — the stored CanonicalProfile would be stale.
    fn merge_profiles_for_resolution(
        existing: &HashMap<String, CanonicalProfile>,
        incoming: &HashMap<String, crate::types::model::ProfileConfig>,
    ) -> HashMap<String, crate::types::model::ProfileConfig> {
        let mut merged: HashMap<String, crate::types::model::ProfileConfig> = existing
            .iter()
            .map(|(name, canon)| {
                (
                    name.clone(),
                    crate::types::model::ProfileConfig {
                        adapter_path: canon.adapter_path.clone(),
                        max_batch_tokens: canon.max_batch_tokens,
                        compute_precision: canon.compute_precision.clone(),
                        adapter_options: canon.adapter_options.clone(),
                        // The stored profile is already inheritance-resolved,
                        // so flatten it: no further extends lookups needed.
                        extends: None,
                    },
                )
            })
            .collect();
        for (name, profile) in incoming {
            merged.insert(name.clone(), profile.clone());
        }
        merged
    }

    fn resolved_profile_from_entry(
        entry: &ModelEntry,
        profile_name: &str,
    ) -> Option<CanonicalProfile> {
        if profile_name == "default" {
            return entry.profile_configs.get("default").cloned();
        }

        let mut resolved =
            entry
                .profile_configs
                .get("default")
                .cloned()
                .unwrap_or(CanonicalProfile {
                    adapter_path: None,
                    max_batch_tokens: None,
                    compute_precision: None,
                    adapter_options: None,
                });
        let profile = entry.profile_configs.get(profile_name)?;

        if profile.adapter_path.is_some() {
            resolved.adapter_path = profile.adapter_path.clone();
        }
        if profile.max_batch_tokens.is_some() {
            resolved.max_batch_tokens = profile.max_batch_tokens;
        }
        if profile.compute_precision.is_some() {
            resolved.compute_precision = profile.compute_precision.clone();
        }
        if profile.adapter_options.is_some() {
            resolved.adapter_options = profile.adapter_options.clone();
        }
        Some(resolved)
    }

    fn adapter_modules_from_entry_profiles(entry: &ModelEntry) -> HashSet<String> {
        entry
            .profile_names
            .iter()
            .filter_map(|profile_name| Self::effective_adapter_path_from_entry(entry, profile_name))
            .filter_map(Self::adapter_module_from_path)
            .collect()
    }

    fn effective_adapter_path_from_profiles<'a>(
        profiles: &'a HashMap<String, crate::types::model::ProfileConfig>,
        profile_name: &str,
    ) -> Option<&'a str> {
        let mut current = Some(profile_name);
        let mut seen = HashSet::new();
        while let Some(name) = current {
            if !seen.insert(name.to_string()) {
                return None;
            }
            let profile = profiles.get(name)?;
            if let Some(adapter_path) = profile.adapter_path.as_deref().filter(|s| !s.is_empty()) {
                return Some(adapter_path);
            }
            current = profile.extends.as_deref();
        }
        None
    }

    fn effective_adapter_path_from_entry<'a>(
        entry: &'a ModelEntry,
        profile_name: &str,
    ) -> Option<&'a str> {
        entry
            .profile_configs
            .get(profile_name)
            .and_then(|profile| profile.adapter_path.as_deref())
            .filter(|s| !s.is_empty())
            .or_else(|| {
                entry
                    .profile_configs
                    .get("default")
                    .and_then(|profile| profile.adapter_path.as_deref())
                    .filter(|s| !s.is_empty())
            })
    }

    fn assign_bundles(entry: &mut ModelEntry, bundles: &HashMap<String, BundleInfo>) {
        if entry.adapter_modules.is_empty() {
            entry.bundles.clear();
            return;
        }
        let mut matching: Vec<(i32, String)> = Vec::new();
        for bundle in bundles.values() {
            let bundle_adapters: HashSet<&str> =
                bundle.adapters.iter().map(|s| s.as_str()).collect();
            let has_overlap = entry
                .adapter_modules
                .iter()
                .any(|a| bundle_adapters.contains(a.as_str()));
            if has_overlap {
                matching.push((bundle.priority, bundle.name.clone()));
            }
        }
        // Break priority ties by bundle name so the default-selected bundle is
        // stable across HashMap iteration order and process restarts.
        matching.sort_by(|(pa, na), (pb, nb)| pa.cmp(pb).then_with(|| na.cmp(nb)));
        entry.bundles = matching.into_iter().map(|(_, name)| name).collect();
    }

    pub fn resolve_bundle(
        &self,
        model: &str,
        bundle_override: Option<&str>,
    ) -> Result<String, ResolveError> {
        self.resolve_bundle_with_engine(model, bundle_override, None)
    }

    /// Engine-aware variant of [`Self::resolve_bundle`].
    ///
    /// When `engine_filter` is `Some(engine)`, the candidate bundle
    /// list for `model` is intersected with bundles whose
    /// `BundleInfo.engine` matches before applying the existing
    /// `(priority, name)` sort. This is what implements the
    /// `X-SIE-Engine` header at the proxy layer; today only
    /// `"pytorch"` is recognised so the filter is effectively a
    /// no-op, but the plumbing is preserved for the day a second
    /// engine lands.
    ///
    /// `engine_filter = None` is the back-compat path used by callers
    /// that don't care about engine selection.
    pub fn resolve_bundle_with_engine(
        &self,
        model: &str,
        bundle_override: Option<&str>,
        engine_filter: Option<&str>,
    ) -> Result<String, ResolveError> {
        let snap = self.snapshot.load();

        // Case-insensitive lookup
        let canonical = Self::canonical_model_name(&snap, model);

        let canonical = match canonical {
            Some(c) => c,
            None => {
                return Err(ResolveError::ModelNotFound(ModelNotFoundError {
                    model: model.to_string(),
                }))
            }
        };

        let model_entry = match snap.models.get(&canonical) {
            Some(e) if !e.bundles.is_empty() => e,
            _ => {
                return Err(ResolveError::ModelNotFound(ModelNotFoundError {
                    model: model.to_string(),
                }))
            }
        };

        let filtered: Vec<String> = match engine_filter {
            Some(eng) => model_entry
                .bundles
                .iter()
                .filter(|b| snap.bundles.get(*b).is_some_and(|info| info.engine == eng))
                .cloned()
                .collect(),
            None => model_entry.bundles.clone(),
        };

        if filtered.is_empty() {
            return Err(ResolveError::BundleConflict(BundleConflictError {
                model: model.to_string(),
                bundle: engine_filter.unwrap_or("").to_string(),
                compatible_bundles: model_entry.bundles.clone(),
            }));
        }

        if let Some(override_bundle) = bundle_override {
            if filtered.contains(&override_bundle.to_string()) {
                return Ok(override_bundle.to_string());
            }
            return Err(ResolveError::BundleConflict(BundleConflictError {
                model: model.to_string(),
                bundle: override_bundle.to_string(),
                compatible_bundles: filtered,
            }));
        }

        Ok(filtered[0].clone())
    }

    pub fn model_exists(&self, model: &str) -> bool {
        let snap = self.snapshot.load();
        snap.model_names_lower.contains_key(&model.to_lowercase())
            || snap.models.contains_key(model)
    }

    /// Resolve a caller-supplied model id to the registry's canonical
    /// stored name, case-insensitively. `model_exists` is already
    /// case-insensitive (it consults `model_names_lower`), so without
    /// canonicalisation `Org/Model`, `org/model`, and `ORG/MODEL` all
    /// pass `model_exists` yet flow downstream as three distinct
    /// strings — and therefore three distinct Prometheus label series
    /// and three distinct dispatch keys. Folding to the canonical name
    /// at the request boundary collapses them to one series / one key.
    ///
    /// Returns `None` when the id is unknown to the registry, so callers
    /// can keep the as-given string for the 404 path.
    pub fn resolve_canonical_model_name(&self, model: &str) -> Option<String> {
        let snap = self.snapshot.load();
        Self::canonical_model_name(&snap, model)
    }

    /// Fast check for "is the registry populated at all?". Used by the
    /// proxy to distinguish "unknown model in a bootstrapped registry"
    /// (→ 404) from "no models configured yet" (→ legacy fallback path).
    pub fn has_any_models(&self) -> bool {
        !self.snapshot.load().models.is_empty()
    }

    pub fn list_models(&self) -> Vec<String> {
        let snap = self.snapshot.load();
        let mut names: Vec<String> = snap.models.keys().cloned().collect();
        names.sort();
        names
    }

    /// Replace the registry's bundle set in-place, then recompute every piece
    /// of derived state that bundles feed into:
    ///
    /// - per-model `bundles` lists (which bundles a model resolves into);
    /// - per-bundle config hashes used by the worker config-skew detector.
    ///
    /// Models are left exactly as they were — only their derived bundle
    /// associations are rebuilt against the new bundle set.
    ///
    /// Used by `state::config_bootstrap` to apply bundles fetched from
    /// `sie-config`'s `GET /v1/configs/bundles` endpoint at startup, so the
    /// gateway no longer needs a filesystem seed of its own. Calling this with
    /// an empty `bundles` list is valid and clears the registry's bundle map
    /// (every subsequent `add_model_config` will then reject every adapter as
    /// unknown — that's the correct behavior when `sie-config` is unreachable
    /// and we have no seed).
    pub fn install_bundles(&self, bundles: Vec<BundleInfo>) {
        let _write = self
            .write_lock
            .lock()
            .expect("ModelRegistry write_lock poisoned");
        let old_snap = self.snapshot.load();
        let mut snap = (**old_snap).clone();

        let mut new_bundles: HashMap<String, BundleInfo> = HashMap::new();
        for bundle in bundles {
            new_bundles.insert(bundle.name.clone(), bundle);
        }

        // Rebuild model→bundle associations against the new bundle set.
        // Mirrors the post-load loop in `reload()` so a freshly-installed
        // bundle set produces identical derived state regardless of whether
        // bundles came from disk or from `sie-config`.
        for model_entry in snap.models.values_mut() {
            if model_entry.adapter_modules.is_empty() {
                model_entry.bundles.clear();
                continue;
            }
            let mut matching: Vec<(i32, String)> = Vec::new();
            for bundle in new_bundles.values() {
                let bundle_adapters: HashSet<&str> =
                    bundle.adapters.iter().map(|s| s.as_str()).collect();
                let has_overlap = model_entry
                    .adapter_modules
                    .iter()
                    .any(|a| bundle_adapters.contains(a.as_str()));
                if has_overlap {
                    matching.push((bundle.priority, bundle.name.clone()));
                }
            }
            // Break priority ties by bundle name so `model_entry.bundles[0]`
            // (the default-selected bundle at route time) is stable across
            // runs — `new_bundles` / bundle iteration comes from a
            // `HashMap`, so without a secondary key equal-priority bundles
            // would shuffle between replicas and between process restarts.
            matching.sort_by(|(pa, na), (pb, nb)| pa.cmp(pb).then_with(|| na.cmp(nb)));
            model_entry.bundles = matching.into_iter().map(|(_, name)| name).collect();
        }

        let bundle_config_hashes = Self::rebuild_bundle_config_hashes(&new_bundles, &snap.models);
        let bundle_pool_config_hashes =
            Self::rebuild_bundle_pool_config_hashes(&new_bundles, &snap.models);

        info!(
            bundles = new_bundles.len(),
            models = snap.models.len(),
            "installed bundles into registry"
        );

        snap.bundles = new_bundles;
        snap.bundle_config_hashes = bundle_config_hashes;
        snap.bundle_pool_config_hashes = bundle_pool_config_hashes;
        self.snapshot.store(Arc::new(snap));
    }

    pub fn list_bundles(&self) -> Vec<String> {
        let snap = self.snapshot.load();
        let mut bundles: Vec<(&String, &BundleInfo)> = snap.bundles.iter().collect();
        bundles.sort_by_key(|(_, b)| b.priority);
        bundles.into_iter().map(|(name, _)| name.clone()).collect()
    }

    pub fn get_bundle_info(&self, bundle: &str) -> Option<BundleInfo> {
        let snap = self.snapshot.load();
        snap.bundles.get(bundle).cloned()
    }

    pub fn get_model_bundles(&self, model: &str) -> Vec<String> {
        let snap = self.snapshot.load();
        let canonical = match Self::canonical_model_name(&snap, model) {
            Some(name) => name,
            None => return Vec::new(),
        };
        snap.models
            .get(&canonical)
            .map(|e| e.bundles.clone())
            .unwrap_or_default()
    }

    pub fn get_model_info(&self, model: &str) -> Option<ModelEntry> {
        let snap = self.snapshot.load();
        let canonical = Self::canonical_model_name(&snap, model)?;
        snap.models.get(&canonical).cloned()
    }

    pub fn get_model_pool_name(&self, model: &str) -> Option<String> {
        let snap = self.snapshot.load();
        let canonical = Self::canonical_model_name(&snap, model)?;
        let entry = snap.models.get(&canonical)?;
        Some(Self::entry_pool_name(entry).to_string())
    }

    pub fn get_model_profile_names(&self, model: &str) -> Vec<String> {
        let mut profiles: Vec<String> = self
            .get_model_info(model)
            .map(|entry| entry.profile_names.into_iter().collect())
            .unwrap_or_default();
        profiles.sort();
        profiles
    }

    /// Look up pre-computed bundle config hash (O(1) HashMap get).
    /// Hash is computed during reload() and add_model_config(), not per-request.
    #[allow(dead_code)]
    pub fn compute_bundle_config_hash(&self, bundle_id: &str) -> String {
        let snap = self.snapshot.load();
        snap.bundle_config_hashes
            .get(bundle_id)
            .cloned()
            .unwrap_or_default()
    }

    pub fn compute_bundle_config_hash_for_pool(&self, bundle_id: &str, pool_name: &str) -> String {
        let snap = self.snapshot.load();
        let pool_name = pool_name.trim().to_lowercase();
        let pool_name = if pool_name.is_empty() {
            DEFAULT_MODEL_POOL.to_string()
        } else {
            pool_name
        };
        snap.bundle_pool_config_hashes
            .get(&(bundle_id.to_string(), pool_name))
            .cloned()
            .unwrap_or_default()
    }

    pub fn install_bundle_config_hashes(&self, hashes: HashMap<String, String>) {
        if hashes.is_empty() {
            return;
        }

        let _guard = self.write_lock.lock().unwrap();
        let mut snap = (**self.snapshot.load()).clone();
        for (bundle_name, hash) in hashes {
            if !snap.bundles.contains_key(&bundle_name) {
                continue;
            }
            if hash.is_empty() {
                snap.bundle_config_hashes.remove(&bundle_name);
            } else {
                snap.bundle_config_hashes.insert(bundle_name, hash);
            }
        }
        self.snapshot.store(Arc::new(snap));
    }

    /// Compute the config hash for a bundle (expensive: sort + JSON + SHA-256).
    /// Called during reload() and add_model_config() to pre-populate the cache.
    fn hash_bundle_config(
        bundle_id: &str,
        bundles: &HashMap<String, BundleInfo>,
        models: &HashMap<String, ModelEntry>,
    ) -> String {
        Self::hash_bundle_config_scoped(bundle_id, bundles, models, None)
    }

    fn hash_bundle_config_for_pool(
        bundle_id: &str,
        bundles: &HashMap<String, BundleInfo>,
        models: &HashMap<String, ModelEntry>,
        pool_name: &str,
    ) -> String {
        Self::hash_bundle_config_scoped(bundle_id, bundles, models, Some(pool_name))
    }

    fn rebuild_bundle_config_hashes(
        bundles: &HashMap<String, BundleInfo>,
        models: &HashMap<String, ModelEntry>,
    ) -> HashMap<String, String> {
        let mut hashes = HashMap::new();
        for bundle_name in bundles.keys() {
            let hash = Self::hash_bundle_config(bundle_name, bundles, models);
            if !hash.is_empty() {
                hashes.insert(bundle_name.clone(), hash);
            }
        }
        hashes
    }

    fn rebuild_bundle_pool_config_hashes(
        bundles: &HashMap<String, BundleInfo>,
        models: &HashMap<String, ModelEntry>,
    ) -> HashMap<(String, String), String> {
        let mut pools_by_bundle: HashMap<String, HashSet<String>> = HashMap::new();
        for (model_name, entry) in models {
            if Self::is_synthetic_profile_variant(model_name, models) {
                continue;
            }
            let pool = Self::entry_pool_name(entry).to_string();
            for bundle in &entry.bundles {
                pools_by_bundle
                    .entry(bundle.clone())
                    .or_default()
                    .insert(pool.clone());
            }
        }

        let mut hashes = HashMap::new();
        for (bundle_name, pools) in pools_by_bundle {
            if !bundles.contains_key(&bundle_name) {
                continue;
            }
            for pool in pools {
                let hash = Self::hash_bundle_config_for_pool(&bundle_name, bundles, models, &pool);
                if !hash.is_empty() {
                    hashes.insert((bundle_name.clone(), pool), hash);
                }
            }
        }
        hashes
    }

    fn hash_bundle_config_scoped(
        bundle_id: &str,
        bundles: &HashMap<String, BundleInfo>,
        models: &HashMap<String, ModelEntry>,
        pool_name: Option<&str>,
    ) -> String {
        let bundle = match bundles.get(bundle_id) {
            Some(b) => b,
            None => return String::new(),
        };
        let normalized_pool = pool_name.map(|p| {
            let p = p.trim().to_lowercase();
            if p.is_empty() {
                DEFAULT_MODEL_POOL.to_string()
            } else {
                p
            }
        });

        let bundle_adapter_set: HashSet<&str> =
            bundle.adapters.iter().map(|s| s.as_str()).collect();

        let mut items: Vec<serde_json::Value> = Vec::new();

        let mut model_names: Vec<&String> = models.keys().collect();
        model_names.sort();

        for model_name in model_names {
            if Self::is_synthetic_profile_variant(model_name, models) {
                continue;
            }

            let model_entry = &models[model_name];
            if normalized_pool
                .as_deref()
                .is_some_and(|pool| Self::entry_pool_name(model_entry) != pool)
            {
                continue;
            }
            if !model_entry.bundles.contains(&bundle_id.to_string()) {
                continue;
            }
            let has_overlap = model_entry
                .adapter_modules
                .iter()
                .any(|a| bundle_adapter_set.contains(a.as_str()));
            if !has_overlap {
                continue;
            }

            let mut profiles_for_hash: Vec<BTreeMap<&str, serde_json::Value>> = Vec::new();
            let mut profile_names: Vec<&String> = model_entry.profile_names.iter().collect();
            profile_names.sort();

            for pname in profile_names {
                if let Some(p_cfg) = model_entry.profile_configs.get(pname) {
                    if let Some(ref adapter_path) = p_cfg.adapter_path {
                        let module = adapter_path.split(':').next().unwrap_or(adapter_path);
                        if !bundle_adapter_set.contains(module) {
                            continue;
                        }
                    }

                    let mut config = BTreeMap::new();
                    config.insert(
                        "adapter_options",
                        serde_json::to_value(&p_cfg.adapter_options)
                            .unwrap_or(serde_json::Value::Null),
                    );
                    config.insert(
                        "adapter_path",
                        serde_json::to_value(&p_cfg.adapter_path)
                            .unwrap_or(serde_json::Value::Null),
                    );
                    config.insert(
                        "compute_precision",
                        serde_json::to_value(&p_cfg.compute_precision)
                            .unwrap_or(serde_json::Value::Null),
                    );
                    config.insert(
                        "max_batch_tokens",
                        serde_json::to_value(p_cfg.max_batch_tokens)
                            .unwrap_or(serde_json::Value::Null),
                    );

                    let mut profile = BTreeMap::new();
                    profile.insert("config", serde_json::to_value(config).unwrap());
                    profile.insert("name", serde_json::Value::String(pname.clone()));
                    profiles_for_hash.push(profile);
                }
            }

            let mut item = BTreeMap::new();
            item.insert("profiles", serde_json::to_value(profiles_for_hash).unwrap());
            item.insert("sie_id", serde_json::Value::String(model_name.clone()));
            items.push(serde_json::to_value(item).unwrap());
        }

        if items.is_empty() {
            return String::new();
        }

        let serialized = serde_json::to_string(&items).unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(serialized.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    fn is_synthetic_profile_variant(
        model_name: &str,
        models: &HashMap<String, ModelEntry>,
    ) -> bool {
        model_name
            .rsplit_once(':')
            .is_some_and(|(base_model_name, profile_name)| {
                !profile_name.is_empty() && models.contains_key(base_model_name)
            })
    }

    pub fn add_model_config(&self, config: ModelConfig) -> Result<AddModelConfigOutcome, String> {
        self.add_model_config_inner(config, false)
    }

    /// Test-only entry point for the per-model authoritative merge branch.
    ///
    /// Production full-snapshot replay uses `replace_model_configs_authoritative`
    /// so removals are applied atomically. This helper keeps the older conflict
    /// override branch covered without exposing it as a runtime API.
    #[cfg(test)]
    pub fn add_model_config_authoritative(
        &self,
        config: ModelConfig,
    ) -> Result<AddModelConfigOutcome, String> {
        self.add_model_config_inner(config, true)
    }

    /// Replace the model surface with the exact exported control-plane set.
    ///
    /// This is the production full-snapshot replay path: it drops models and
    /// profile variants that disappeared from `sie-config` instead of overlaying
    /// exported rows onto the existing snapshot. Use this only for
    /// `/v1/configs/export`, never for live NATS deltas.
    pub fn replace_model_configs_authoritative(
        &self,
        configs: Vec<ModelConfig>,
    ) -> Result<usize, String> {
        let _write = self
            .write_lock
            .lock()
            .expect("ModelRegistry write_lock poisoned");
        let old_snap = self.snapshot.load();
        let mut new_models: HashMap<String, ModelEntry> = HashMap::new();
        let mut new_model_names_lower: HashMap<String, String> = HashMap::new();
        let applied = configs.len();

        for config in configs {
            let sie_id = &config.name;
            if sie_id.is_empty() {
                return Err("Missing required field: name/sie_id".to_string());
            }
            if sie_id.chars().any(|c| c.is_control()) {
                return Err(format!(
                    "Invalid model id: contains control character (sie_id={:?})",
                    sie_id
                ));
            }
            if config.profiles.is_empty() {
                return Err("Missing required field: profiles".to_string());
            }

            let mut adapter_modules: HashSet<String> = HashSet::new();
            for (profile_name, profile) in &config.profiles {
                if let Some(adapter_path) =
                    Self::effective_adapter_path_from_profiles(&config.profiles, profile_name)
                {
                    if let Some(module) = Self::adapter_module_from_path(adapter_path) {
                        adapter_modules.insert(module);
                    }
                } else if profile.extends.is_none() {
                    return Err(format!("Profile '{}' missing adapter_path", profile_name));
                }
            }

            let all_bundle_adapters: HashSet<String> = old_snap
                .bundles
                .values()
                .flat_map(|b| b.adapters.iter().cloned())
                .collect();
            let unroutable: Vec<&String> = adapter_modules
                .iter()
                .filter(|a| !all_bundle_adapters.contains(a.as_str()))
                .collect();
            if !unroutable.is_empty() {
                return Err(format!(
                    "Adapter(s) not in any known bundle: {}",
                    unroutable
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }

            for mut entry in Self::expand_model_config_into_profile_variants(&config)? {
                Self::assign_bundles(&mut entry, &old_snap.bundles);
                new_model_names_lower.insert(entry.name.to_lowercase(), entry.name.clone());
                new_models.insert(entry.name.clone(), entry);
            }
        }

        let bundle_config_hashes =
            Self::rebuild_bundle_config_hashes(&old_snap.bundles, &new_models);
        let bundle_pool_config_hashes =
            Self::rebuild_bundle_pool_config_hashes(&old_snap.bundles, &new_models);

        info!(
            old_models = old_snap.models.len(),
            models = new_models.len(),
            bundles = old_snap.bundles.len(),
            "replaced model configs from authoritative export"
        );

        self.snapshot.store(Arc::new(RegistrySnapshot {
            bundles: old_snap.bundles.clone(),
            models: new_models,
            model_names_lower: new_model_names_lower,
            bundle_config_hashes,
            bundle_pool_config_hashes,
        }));

        Ok(applied)
    }

    fn add_model_config_inner(
        &self,
        config: ModelConfig,
        authoritative: bool,
    ) -> Result<AddModelConfigOutcome, String> {
        // Serialize mutators. `ArcSwap` gives us lock-free reads but no CAS
        // on the write path — without this lock, two concurrent callers can
        // both `load()` the same base snapshot, both build a derived copy
        // with their own mutation, and the second `store()` silently drops
        // the first caller's changes. In practice this raced the NATS
        // subscription task against the bootstrap retry task.
        let _write = self
            .write_lock
            .lock()
            .expect("ModelRegistry write_lock poisoned");
        let old_snap = self.snapshot.load();
        let mut snap = (**old_snap).clone();

        let sie_id = &config.name;
        if sie_id.is_empty() {
            return Err("Missing required field: name/sie_id".to_string());
        }

        // Reject model IDs containing control characters (newlines, carriage
        // returns, NULs, etc.). These can break SSE ``data:`` framing and
        // log structured output if propagated downstream; gating at
        // registration time keeps every downstream consumer simple.
        if sie_id.chars().any(|c| c.is_control()) {
            return Err(format!(
                "Invalid model id: contains control character (sie_id={:?})",
                sie_id
            ));
        }

        if config.profiles.is_empty() {
            return Err("Missing required field: profiles".to_string());
        }

        let mut new_adapter_modules: HashSet<String> = HashSet::new();
        for (profile_name, profile) in &config.profiles {
            if let Some(adapter_path) =
                Self::effective_adapter_path_from_profiles(&config.profiles, profile_name)
            {
                if let Some(module) = Self::adapter_module_from_path(adapter_path) {
                    new_adapter_modules.insert(module);
                }
            } else if profile.extends.is_none() {
                return Err(format!("Profile '{}' missing adapter_path", profile_name));
            }
        }

        // Validate adapter modules are routable
        let all_bundle_adapters: HashSet<String> = snap
            .bundles
            .values()
            .flat_map(|b| b.adapters.iter().cloned())
            .collect();

        let unroutable: Vec<&String> = new_adapter_modules
            .iter()
            .filter(|a| !all_bundle_adapters.contains(a.as_str()))
            .collect();

        if !unroutable.is_empty() {
            return Err(format!(
                "Adapter(s) not in any known bundle: {}",
                unroutable
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }

        let mut created_profiles: Vec<String> = Vec::new();
        let mut skipped_profiles: Vec<String> = Vec::new();
        let mut overridden_profiles: Vec<String> = Vec::new();
        let incoming_pool = Self::normalize_model_pool(config.pool.as_deref())?;

        if let Some(existing) = snap.models.get_mut(sie_id) {
            if config.pool.is_some() && existing.pool != incoming_pool {
                return Err(format!(
                    "Pool on model '{}' already exists with different value (append-only)",
                    sie_id
                ));
            }
            // Append-only by default: add new profiles, skip identical,
            // reject conflicts. The test-only authoritative branch preserves
            // the per-model overwrite behavior that predates full-snapshot
            // replacement.
            //
            // Resolution map: a delta update can carry a profile that
            // ``extends`` a profile only present in ``existing.profile_configs``
            // (e.g. an incoming ``a100-40gb`` extending an already-stored
            // ``default``). Build a temporary ``ProfileConfig`` map combining
            // already-stored canonical profiles with the incoming ones —
            // incoming wins on collisions — and resolve against that.
            let merged_profiles =
                Self::merge_profiles_for_resolution(&existing.profile_configs, &config.profiles);

            for profile_name in config.profiles.keys() {
                if existing.profile_names.contains(profile_name) {
                    let incoming =
                        Self::resolved_profile_from_profiles(&merged_profiles, profile_name)?;
                    if let Some(stored) = existing.profile_configs.get(profile_name) {
                        if stored != &incoming {
                            if !authoritative {
                                return Err(format!(
                                    "Profile '{}' on model '{}' already exists with different config (append-only)",
                                    profile_name, sie_id
                                ));
                            }
                            warn!(
                                model = %sie_id,
                                profile = %profile_name,
                                old_adapter_path = ?stored.adapter_path,
                                new_adapter_path = ?incoming.adapter_path,
                                old_max_batch_tokens = ?stored.max_batch_tokens,
                                new_max_batch_tokens = ?incoming.max_batch_tokens,
                                old_compute_precision = ?stored.compute_precision,
                                new_compute_precision = ?incoming.compute_precision,
                                "overriding existing profile config from authoritative source (sie-config)",
                            );
                            overridden_profiles.push(profile_name.clone());
                            continue;
                        }
                    }
                    skipped_profiles.push(profile_name.clone());
                } else {
                    created_profiles.push(profile_name.clone());
                }
            }

            for pname in &created_profiles {
                existing.profile_names.insert(pname.clone());
                existing.profile_configs.insert(
                    pname.clone(),
                    Self::resolved_profile_from_profiles(&merged_profiles, pname)?,
                );
            }
            // Authoritative overrides: replace the stored CanonicalProfile.
            for pname in &overridden_profiles {
                existing.profile_configs.insert(
                    pname.clone(),
                    Self::resolved_profile_from_profiles(&merged_profiles, pname)?,
                );
            }
            existing.adapter_modules = Self::adapter_modules_from_entry_profiles(existing);
            Self::merge_model_info_extras(&mut existing.info_extras, &config, &merged_profiles);
        } else {
            // New model
            let mut profile_names: HashSet<String> = HashSet::new();
            let mut profile_configs: HashMap<String, CanonicalProfile> = HashMap::new();

            for pname in config.profiles.keys() {
                profile_names.insert(pname.clone());
                profile_configs.insert(
                    pname.clone(),
                    Self::resolved_profile_from_profiles(&config.profiles, pname)?,
                );
            }

            created_profiles = config.profiles.keys().cloned().collect();

            let info_extras = ModelInfoExtras::from_model_config(&config);
            snap.models.insert(
                sie_id.clone(),
                ModelEntry {
                    name: sie_id.clone(),
                    pool: incoming_pool,
                    bundles: Vec::new(),
                    adapter_modules: new_adapter_modules,
                    profile_names,
                    profile_configs,
                    info_extras,
                },
            );
            snap.model_names_lower
                .insert(sie_id.to_lowercase(), sie_id.clone());
        }

        let variant_prefix = format!("{sie_id}:");
        let mut affected_bundles: Vec<String> = snap
            .models
            .iter()
            .filter(|(name, _)| *name == sie_id || name.starts_with(&variant_prefix))
            .flat_map(|(_, entry)| entry.bundles.clone())
            .collect();
        let mut affected_model_names = vec![sie_id.clone()];
        if let Some(base_entry) = snap.models.get(sie_id).cloned() {
            let old_variants: Vec<String> = snap
                .models
                .keys()
                .filter(|name| name.starts_with(&variant_prefix))
                .cloned()
                .collect();
            for name in old_variants {
                snap.models.remove(&name);
                snap.model_names_lower.remove(&name.to_lowercase());
            }

            for mut entry in Self::expand_model_entry_into_profile_variants(&base_entry)? {
                Self::assign_bundles(&mut entry, &snap.bundles);
                affected_model_names.push(entry.name.clone());
                snap.model_names_lower
                    .insert(entry.name.to_lowercase(), entry.name.clone());
                snap.models.insert(entry.name.clone(), entry);
            }
        }

        affected_bundles.extend(
            affected_model_names
                .iter()
                .filter_map(|name| snap.models.get(name))
                .flat_map(|e| e.bundles.clone()),
        );
        affected_bundles.sort();
        affected_bundles.dedup();

        // Recompute config hashes for affected bundles
        for bundle_name in &affected_bundles {
            let hash = Self::hash_bundle_config(bundle_name, &snap.bundles, &snap.models);
            if hash.is_empty() {
                snap.bundle_config_hashes.remove(bundle_name);
            } else {
                snap.bundle_config_hashes.insert(bundle_name.clone(), hash);
            }
        }
        snap.bundle_pool_config_hashes =
            Self::rebuild_bundle_pool_config_hashes(&snap.bundles, &snap.models);

        info!(
            model = %sie_id,
            created = ?created_profiles,
            skipped = ?skipped_profiles,
            overridden = ?overridden_profiles,
            bundles = ?affected_bundles,
            "added model config"
        );

        self.snapshot.store(Arc::new(snap));

        Ok((created_profiles, skipped_profiles, affected_bundles))
    }

    fn merge_model_info_extras(
        existing: &mut ModelInfoExtras,
        config: &ModelConfig,
        merged_profiles: &HashMap<String, ProfileConfig>,
    ) {
        let touches_profiles = !config.profiles.is_empty();
        if config.inputs.is_none()
            && config.tasks.is_none()
            && config.max_sequence_length.is_none()
            && !touches_profiles
        {
            return;
        }

        let refreshed = ModelInfoExtras::from_model_config(config);
        if config.inputs.is_some() {
            existing.inputs = refreshed.inputs;
        }
        if config.tasks.is_some() {
            existing.outputs = refreshed.outputs;
            existing.dims = refreshed.dims;
            existing.max_output_tokens = refreshed.max_output_tokens;
            // M10 follow-up: grammar+tools capabilities are derived
            // from ``tasks.generate.capabilities`` on the model-level
            // config — refresh here so a delta-update can rescind or
            // tighten the advertised set without a full reload.
            existing.grammar_capabilities = refreshed.grammar_capabilities;
            existing.tools_supported = refreshed.tools_supported;
        }
        if config.max_sequence_length.is_some() {
            existing.max_sequence_length = refreshed.max_sequence_length;
        }
        // M10 follow-up: ``lora_adapters`` and ``profile_lora_adapters``
        // are derived from ``profiles.*.adapter_options.loadtime.lora_paths``.
        // A delta config typically only carries the profiles being added
        // or modified, so recomputing from ``config`` alone would drop
        // adapters declared on profiles that weren't part of the delta.
        // We rebuild from the full ``merged_profiles`` set (existing +
        // incoming) instead so the published capability map matches what
        // the worker has actually loaded.
        if touches_profiles {
            let mut merged_config = config.clone();
            merged_config.profiles = merged_profiles.clone();
            let lora_refreshed = ModelInfoExtras::from_model_config(&merged_config);
            existing.lora_adapters = lora_refreshed.lora_adapters;
            existing.profile_lora_adapters = lora_refreshed.profile_lora_adapters;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn create_test_dirs() -> (TempDir, PathBuf, PathBuf) {
        let dir = TempDir::new().unwrap();
        let bundles_dir = dir.path().join("bundles");
        let models_dir = dir.path().join("models");
        fs::create_dir_all(&bundles_dir).unwrap();
        fs::create_dir_all(&models_dir).unwrap();
        (dir, bundles_dir, models_dir)
    }

    fn profile(
        adapter_path: Option<&str>,
        max_batch_tokens: Option<u32>,
        extends: Option<&str>,
    ) -> crate::types::model::ProfileConfig {
        crate::types::model::ProfileConfig {
            adapter_path: adapter_path.map(str::to_string),
            max_batch_tokens,
            compute_precision: None,
            adapter_options: None,
            extends: extends.map(str::to_string),
        }
    }

    #[test]
    fn test_empty_registry() {
        let (_dir, bundles_dir, models_dir) = create_test_dirs();
        let registry = ModelRegistry::new(&bundles_dir, &models_dir, true);
        assert!(registry.list_models().is_empty());
        assert!(registry.list_bundles().is_empty());
        assert!(!registry.has_any_models());
    }

    #[test]
    fn test_has_any_models_flips_after_add() {
        use crate::types::model::{ModelConfig, ProfileConfig};
        use std::collections::HashMap as StdHashMap;

        let (_dir, bundles_dir, models_dir) = create_test_dirs();
        fs::write(
            bundles_dir.join("default.yaml"),
            "name: default\npriority: 10\nadapters:\n  - sie_server.adapters.sentence_transformer\n",
        )
        .unwrap();
        let registry = ModelRegistry::new(&bundles_dir, &models_dir, true);
        assert!(
            !registry.has_any_models(),
            "empty registry must report no models"
        );

        let mut profiles = StdHashMap::new();
        profiles.insert(
            "default".to_string(),
            ProfileConfig {
                adapter_path: Some("sie_server.adapters.sentence_transformer:Adapter".to_string()),
                max_batch_tokens: Some(4096),
                compute_precision: None,
                adapter_options: None,
                extends: None,
            },
        );
        registry
            .add_model_config(ModelConfig {
                name: "org/x".to_string(),
                adapter_module: None,
                default_bundle: None,
                pool: None,
                profiles,
                inputs: None,
                max_sequence_length: None,
                tasks: None,
            })
            .unwrap();
        assert!(registry.has_any_models(), "must report populated after add");
    }

    #[test]
    fn test_add_model_config_rejects_invalid_profile_name_live_path() {
        // H-medium: the live delta path (`add_model_config` →
        // `expand_model_entry_into_profile_variants`) must enforce the
        // same profile-name allowlist as the file-load path, so a
        // control-plane delta carrying e.g. `bad/name` is rejected
        // instead of minting a phantom `{base}:{bad/name}` registry entry.
        use crate::types::model::{ModelConfig, ProfileConfig};
        use std::collections::HashMap as StdHashMap;

        let (_dir, bundles_dir, models_dir) = create_test_dirs();
        fs::write(
            bundles_dir.join("default.yaml"),
            "name: default\npriority: 10\nadapters:\n  - sie_server.adapters.sentence_transformer\n",
        )
        .unwrap();
        let registry = ModelRegistry::new(&bundles_dir, &models_dir, true);

        let mut profiles = StdHashMap::new();
        profiles.insert(
            "default".to_string(),
            ProfileConfig {
                adapter_path: Some("sie_server.adapters.sentence_transformer:Adapter".to_string()),
                max_batch_tokens: Some(4096),
                compute_precision: None,
                adapter_options: None,
                extends: None,
            },
        );
        // Invalid profile name: contains '/', which would shadow a base
        // model id when concatenated as `{base}:{profile}`.
        profiles.insert(
            "bad/name".to_string(),
            ProfileConfig {
                adapter_path: Some("sie_server.adapters.sentence_transformer:Adapter".to_string()),
                max_batch_tokens: Some(4096),
                compute_precision: None,
                adapter_options: None,
                extends: None,
            },
        );

        let err = registry
            .add_model_config(ModelConfig {
                name: "org/x".to_string(),
                adapter_module: None,
                default_bundle: None,
                pool: None,
                profiles,
                inputs: None,
                max_sequence_length: None,
                tasks: None,
            })
            .expect_err("invalid profile name must be rejected on the live path");
        assert!(
            err.contains("profile name"),
            "error should mention the profile-name rule, got: {err}"
        );
        // And the phantom variant must NOT have been registered.
        assert!(!registry.model_exists("org/x:bad/name"));
    }

    #[test]
    fn test_reload_expands_non_default_profiles_as_variants() {
        let (_dir, bundles_dir, models_dir) = create_test_dirs();
        fs::write(
            bundles_dir.join("sglang.yaml"),
            "name: sglang\npriority: 20\nadapters:\n  - sie_server.adapters.sglang.generation\n",
        )
        .unwrap();
        fs::write(
            models_dir.join("Qwen__Qwen3.5-4B.yaml"),
            r#"
sie_id: Qwen/Qwen3.5-4B
profiles:
  default:
    adapter_path: sie_server.adapters.sglang.generation:SGLangGenerationAdapter
    max_batch_tokens: 8192
  a100-40gb:
    extends: default
    max_batch_tokens: 32768
"#,
        )
        .unwrap();

        let registry = ModelRegistry::new(&bundles_dir, &models_dir, true);

        assert!(registry.model_exists("Qwen/Qwen3.5-4B"));
        assert!(registry.model_exists("Qwen/Qwen3.5-4B:a100-40gb"));
        assert_eq!(
            registry
                .resolve_bundle("Qwen/Qwen3.5-4B:a100-40gb", None)
                .unwrap(),
            "sglang"
        );
        assert_eq!(
            registry
                .get_model_info("Qwen/Qwen3.5-4B:a100-40gb")
                .unwrap()
                .profile_names,
            HashSet::from(["default".to_string()])
        );
        let variant = registry
            .get_model_info("Qwen/Qwen3.5-4B:a100-40gb")
            .unwrap();
        let default_profile = variant.profile_configs.get("default").unwrap();
        assert_eq!(
            default_profile.adapter_path.as_deref(),
            Some("sie_server.adapters.sglang.generation:SGLangGenerationAdapter")
        );
        assert_eq!(default_profile.max_batch_tokens, Some(32768));
    }

    #[test]
    fn test_variant_lora_adapters_narrowed_to_variant_profile() {
        // M10 regression: a non-default profile variant is materialized
        // as its own ``ModelEntry`` (``{base}:{profile}``). That
        // variant's ``info_extras`` must advertise / validate against
        // ONLY the suffix-profile's LoRA adapters — not the union the
        // base entry carries. Otherwise the gateway accepts an adapter
        // that's only configured for a sibling profile, and the worker
        // fails opaquely instead of returning a clean 400.
        let (_dir, bundles_dir, models_dir) = create_test_dirs();
        fs::write(
            bundles_dir.join("sglang.yaml"),
            "name: sglang\npriority: 20\nadapters:\n  - sie_server.adapters.sglang.generation\n",
        )
        .unwrap();
        fs::write(
            models_dir.join("acme__multi.yaml"),
            r#"
sie_id: acme/multi
profiles:
  default:
    adapter_path: sie_server.adapters.sglang.generation:SGLangGenerationAdapter
    adapter_options:
      loadtime:
        lora_paths:
          a1: acme/a1
          a2: acme/a2
  a100:
    adapter_path: sie_server.adapters.sglang.generation:SGLangGenerationAdapter
    adapter_options:
      loadtime:
        lora_paths:
          b1: acme/b1
"#,
        )
        .unwrap();

        let registry = ModelRegistry::new(&bundles_dir, &models_dir, true);

        // Base entry carries the union (back-compat) and the full
        // per-profile breakdown.
        let base = registry.get_model_info("acme/multi").unwrap();
        let mut base_union = base.info_extras.lora_adapters.clone().unwrap();
        base_union.sort();
        assert_eq!(
            base_union,
            vec!["a1".to_string(), "a2".to_string(), "b1".to_string()]
        );
        // Variant entry only sees its own profile's adapters, both in
        // the union summary AND in the per-profile map (keyed under
        // ``"default"`` because the variant's profile_configs collapses
        // the resolved profile to that name).
        let variant = registry.get_model_info("acme/multi:a100").unwrap();
        assert_eq!(
            variant.info_extras.lora_adapters,
            Some(vec!["b1".to_string()]),
            "variant lora_adapters union must be narrowed to its profile"
        );
        assert_eq!(
            variant.lora_adapters_for_profile("default"),
            Some(&vec!["b1".to_string()]),
            "variant lora_adapters_for_profile(default) must yield the variant's profile adapters"
        );
        // Cross-check: requesting the base's adapter through the
        // variant would now reject (not in the variant's list).
        assert!(!variant
            .lora_adapters_for_profile("default")
            .map(|v| v.iter().any(|s| s == "a1"))
            .unwrap_or(false));
    }

    #[test]
    fn test_reload_rejects_model_with_missing_profile_parent() {
        let (_dir, bundles_dir, models_dir) = create_test_dirs();
        fs::write(
            bundles_dir.join("default.yaml"),
            "name: default\npriority: 10\nadapters:\n  - module\n",
        )
        .unwrap();
        fs::write(
            models_dir.join("broken.yaml"),
            r#"
name: org/broken
profiles:
  child:
    extends: missing
    max_batch_tokens: 8192
"#,
        )
        .unwrap();

        let registry = ModelRegistry::new(&bundles_dir, &models_dir, true);

        assert!(!registry.model_exists("org/broken"));
        assert!(registry.list_models().is_empty());
    }

    #[test]
    fn test_add_model_config_rejects_broken_profile_parent() {
        let (_dir, bundles_dir, models_dir) = create_test_dirs();
        fs::write(
            bundles_dir.join("default.yaml"),
            "name: default\npriority: 10\nadapters:\n  - module\n",
        )
        .unwrap();
        let registry = ModelRegistry::new(&bundles_dir, &models_dir, true);
        let mut profiles = HashMap::new();
        profiles.insert(
            "child".to_string(),
            profile(None, Some(8192), Some("missing")),
        );

        let err = registry
            .add_model_config(ModelConfig {
                name: "org/broken".to_string(),
                adapter_module: None,
                default_bundle: None,
                pool: None,
                profiles,
                inputs: None,
                max_sequence_length: None,
                tasks: None,
            })
            .expect_err("broken profile inheritance must reject the config");

        assert!(err.contains("Profile 'child'"));
        assert!(err.contains("missing parent or an inheritance cycle"));
        assert!(!registry.model_exists("org/broken"));
    }

    #[test]
    fn test_concurrent_add_model_config_does_not_lose_updates() {
        // Guards the `write_lock` around `snapshot.load() + clone() +
        // store()` in `add_model_config`. Without the lock, concurrent
        // writers can both load the same base snapshot and one of the
        // updates gets silently dropped. Every model added by every worker
        // must end up in the registry.
        use crate::types::model::{ModelConfig, ProfileConfig};
        use std::collections::HashMap as StdHashMap;
        use std::thread;

        let (_dir, bundles_dir, models_dir) = create_test_dirs();
        fs::write(
            bundles_dir.join("default.yaml"),
            "name: default\npriority: 10\nadapters:\n  - sie_server.adapters.sentence_transformer\n",
        )
        .unwrap();
        let registry = Arc::new(ModelRegistry::new(&bundles_dir, &models_dir, true));

        let workers = 16;
        let adds_per_worker = 25;
        let mut handles = Vec::with_capacity(workers);
        for w in 0..workers {
            let registry = Arc::clone(&registry);
            handles.push(thread::spawn(move || {
                for i in 0..adds_per_worker {
                    let name = format!("w{}/m{}", w, i);
                    let mut profiles = StdHashMap::new();
                    profiles.insert(
                        "default".to_string(),
                        ProfileConfig {
                            adapter_path: Some(
                                "sie_server.adapters.sentence_transformer:Adapter".to_string(),
                            ),
                            max_batch_tokens: Some(4096),
                            compute_precision: None,
                            adapter_options: None,
                            extends: None,
                        },
                    );
                    let cfg = ModelConfig {
                        name: name.clone(),
                        adapter_module: None,
                        default_bundle: None,
                        pool: None,
                        profiles,
                        inputs: None,
                        max_sequence_length: None,
                        tasks: None,
                    };
                    registry.add_model_config(cfg).expect("add must succeed");
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        let models = registry.list_models();
        assert_eq!(
            models.len(),
            workers * adds_per_worker,
            "every parallel insert must be reflected in the registry"
        );
    }

    #[test]
    fn test_load_bundle() {
        let (_dir, bundles_dir, models_dir) = create_test_dirs();

        fs::write(
            bundles_dir.join("default.yaml"),
            r#"
name: default
priority: 10
adapters:
  - sie_server.adapters.sentence_transformer
  - sie_server.adapters.cross_encoder
default: true
"#,
        )
        .unwrap();

        let registry = ModelRegistry::new(&bundles_dir, &models_dir, true);
        let bundles = registry.list_bundles();
        assert_eq!(bundles.len(), 1);
        assert_eq!(bundles[0], "default");

        let info = registry.get_bundle_info("default").unwrap();
        assert_eq!(info.priority, 10);
        assert_eq!(info.adapters.len(), 2);
    }

    #[test]
    fn test_load_model_and_resolve() {
        let (_dir, bundles_dir, models_dir) = create_test_dirs();

        fs::write(
            bundles_dir.join("default.yaml"),
            r#"
name: default
priority: 10
adapters:
  - sie_server.adapters.sentence_transformer
"#,
        )
        .unwrap();

        fs::write(
            models_dir.join("bge-m3.yaml"),
            r#"
name: BAAI/bge-m3
profiles:
  default:
    adapter_path: "sie_server.adapters.sentence_transformer:SentenceTransformerAdapter"
    max_batch_tokens: 4096
"#,
        )
        .unwrap();

        let registry = ModelRegistry::new(&bundles_dir, &models_dir, true);
        assert!(registry.model_exists("BAAI/bge-m3"));
        assert!(registry.model_exists("baai/bge-m3")); // case insensitive

        let bundle = registry.resolve_bundle("BAAI/bge-m3", None).unwrap();
        assert_eq!(bundle, "default");
    }

    #[test]
    fn test_resolve_model_not_found() {
        let (_dir, bundles_dir, models_dir) = create_test_dirs();
        let registry = ModelRegistry::new(&bundles_dir, &models_dir, true);

        let result = registry.resolve_bundle("nonexistent/model", None);
        assert!(matches!(result, Err(ResolveError::ModelNotFound(_))));
    }

    #[test]
    fn test_resolve_bundle_conflict() {
        let (_dir, bundles_dir, models_dir) = create_test_dirs();

        fs::write(
            bundles_dir.join("default.yaml"),
            r#"
name: default
priority: 10
adapters:
  - sie_server.adapters.sentence_transformer
"#,
        )
        .unwrap();

        fs::write(
            models_dir.join("bge-m3.yaml"),
            r#"
name: BAAI/bge-m3
profiles:
  default:
    adapter_path: "sie_server.adapters.sentence_transformer:SentenceTransformerAdapter"
"#,
        )
        .unwrap();

        let registry = ModelRegistry::new(&bundles_dir, &models_dir, true);
        let result = registry.resolve_bundle("BAAI/bge-m3", Some("nonexistent"));
        assert!(matches!(result, Err(ResolveError::BundleConflict(_))));
    }

    #[test]
    fn test_precision_bundle_override_routes_or_conflicts() {
        // Validates the premise behind bundle-carrying job aliases (proxy.rs
        // resolve_model_spec_with_aliases): one base model can be registered
        // under TWO precision-labeled bundles that share the same generation
        // adapter, so a `<bundle>:/model` override (what a `sql` alias expands
        // to) actually routes to the chosen precision — and an unregistered
        // bundle 409s rather than silently falling back to the FP8 default.
        let (_dir, bundles_dir, models_dir) = create_test_dirs();

        // Two bundles, same generation adapter → both match the model.
        for name in ["fp8-pytorch", "bf16-pytorch"] {
            fs::write(
                bundles_dir.join(format!("{name}.yaml")),
                format!(
                    r#"
name: {name}
priority: 10
adapters:
  - sie_server.adapters.sglang.generation
"#
                ),
            )
            .unwrap();
        }

        fs::write(
            models_dir.join("qwen3-27b.yaml"),
            r#"
name: Qwen/Qwen3.6-27B
profiles:
  default:
    adapter_path: "sie_server.adapters.sglang.generation:SGLangGenerationAdapter"
"#,
        )
        .unwrap();

        let registry = ModelRegistry::new(&bundles_dir, &models_dir, true);

        // The model lives in BOTH bundles → an operator can deploy a BF16
        // variant under its own bundle (the deployment story the ADR describes).
        assert_eq!(
            registry
                .resolve_bundle("Qwen/Qwen3.6-27B", Some("bf16-pytorch"))
                .unwrap(),
            "bf16-pytorch",
        );
        assert_eq!(
            registry
                .resolve_bundle("Qwen/Qwen3.6-27B", Some("fp8-pytorch"))
                .unwrap(),
            "fp8-pytorch",
        );
        // An alias pointing at a bundle the model is NOT registered under is a
        // hard BundleConflict (409), not a silent fallback — the failure mode
        // an operator must avoid when wiring SIE_GATEWAY_MODEL_ALIASES.
        assert!(matches!(
            registry.resolve_bundle("Qwen/Qwen3.6-27B", Some("no-such-bundle")),
            Err(ResolveError::BundleConflict(_)),
        ));
    }

    #[test]
    fn test_bundle_priority_ordering() {
        let (_dir, bundles_dir, models_dir) = create_test_dirs();

        fs::write(
            bundles_dir.join("high.yaml"),
            r#"
name: high
priority: 50
adapters:
  - sie_server.adapters.sentence_transformer
"#,
        )
        .unwrap();

        fs::write(
            bundles_dir.join("low.yaml"),
            r#"
name: low
priority: 5
adapters:
  - sie_server.adapters.sentence_transformer
"#,
        )
        .unwrap();

        fs::write(
            models_dir.join("bge-m3.yaml"),
            r#"
name: BAAI/bge-m3
profiles:
  default:
    adapter_path: "sie_server.adapters.sentence_transformer:SentenceTransformerAdapter"
"#,
        )
        .unwrap();

        let registry = ModelRegistry::new(&bundles_dir, &models_dir, true);

        // "low" has priority 5, "high" has 50. Lower priority number = higher priority.
        let bundle = registry.resolve_bundle("BAAI/bge-m3", None).unwrap();
        assert_eq!(bundle, "low");
    }

    #[test]
    fn test_add_model_config() {
        let (_dir, bundles_dir, models_dir) = create_test_dirs();

        fs::write(
            bundles_dir.join("default.yaml"),
            r#"
name: default
priority: 10
adapters:
  - sie_server.adapters.sentence_transformer
"#,
        )
        .unwrap();

        let registry = ModelRegistry::new(&bundles_dir, &models_dir, true);

        let config = ModelConfig {
            name: "test/model".to_string(),
            adapter_module: None,
            default_bundle: None,
            pool: None,
            profiles: {
                let mut m = HashMap::new();
                m.insert(
                    "default".to_string(),
                    crate::types::model::ProfileConfig {
                        adapter_path: Some(
                            "sie_server.adapters.sentence_transformer:SentenceTransformerAdapter"
                                .to_string(),
                        ),
                        max_batch_tokens: Some(4096),
                        compute_precision: None,
                        adapter_options: None,
                        extends: None,
                    },
                );
                m
            },
            inputs: None,
            max_sequence_length: None,
            tasks: None,
        };

        let (created, skipped, bundles) = registry.add_model_config(config).unwrap();
        assert_eq!(created, vec!["default"]);
        assert!(skipped.is_empty());
        assert_eq!(bundles, vec!["default"]);

        assert!(registry.model_exists("test/model"));
        let resolved = registry.resolve_bundle("test/model", None).unwrap();
        assert_eq!(resolved, "default");
    }

    #[test]
    fn test_add_model_config_refreshes_existing_model_info_extras() {
        let (_dir, bundles_dir, models_dir) = create_test_dirs();

        fs::write(
            bundles_dir.join("default.yaml"),
            r#"
name: default
priority: 10
adapters:
  - module
"#,
        )
        .unwrap();

        let registry = ModelRegistry::new(&bundles_dir, &models_dir, true);
        let mut profiles = HashMap::new();
        profiles.insert(
            "default".to_string(),
            crate::types::model::ProfileConfig {
                adapter_path: Some("module:Adapter".to_string()),
                max_batch_tokens: Some(4096),
                compute_precision: None,
                adapter_options: None,
                extends: None,
            },
        );
        registry
            .add_model_config(ModelConfig {
                name: "test/model".to_string(),
                adapter_module: None,
                default_bundle: None,
                pool: None,
                profiles,
                inputs: None,
                max_sequence_length: Some(512),
                tasks: Some(
                    serde_yaml::from_str(
                        r#"
encode:
  dense:
    dim: 384
"#,
                    )
                    .unwrap(),
                ),
            })
            .unwrap();

        let first = registry.get_model_info("test/model").unwrap();
        assert_eq!(first.info_extras.outputs, vec!["dense"]);
        assert_eq!(first.info_extras.dims["dense"], 384);
        assert_eq!(first.info_extras.max_sequence_length, Some(512));

        let mut profiles = HashMap::new();
        profiles.insert(
            "default".to_string(),
            crate::types::model::ProfileConfig {
                adapter_path: Some("module:Adapter".to_string()),
                max_batch_tokens: Some(4096),
                compute_precision: None,
                adapter_options: None,
                extends: None,
            },
        );
        profiles.insert(
            "alt".to_string(),
            crate::types::model::ProfileConfig {
                adapter_path: Some("module:Adapter".to_string()),
                max_batch_tokens: Some(2048),
                compute_precision: None,
                adapter_options: None,
                extends: None,
            },
        );

        registry
            .add_model_config(ModelConfig {
                name: "test/model".to_string(),
                adapter_module: None,
                default_bundle: None,
                pool: None,
                profiles,
                inputs: None,
                max_sequence_length: Some(1024),
                tasks: Some(
                    serde_yaml::from_str(
                        r#"
encode:
  sparse:
    dim: 30000
"#,
                    )
                    .unwrap(),
                ),
            })
            .unwrap();

        let refreshed = registry.get_model_info("test/model").unwrap();
        assert!(refreshed.profile_names.contains("alt"));
        assert_eq!(refreshed.info_extras.outputs, vec!["sparse"]);
        assert_eq!(refreshed.info_extras.dims["sparse"], 30000);
        assert_eq!(refreshed.info_extras.max_sequence_length, Some(1024));
    }

    #[test]
    fn test_add_model_config_preserves_info_extras_for_profile_only_update() {
        let (_dir, bundles_dir, models_dir) = create_test_dirs();

        fs::write(
            bundles_dir.join("default.yaml"),
            r#"
name: default
priority: 10
adapters:
  - module
"#,
        )
        .unwrap();

        let registry = ModelRegistry::new(&bundles_dir, &models_dir, true);
        let mut profiles = HashMap::new();
        profiles.insert(
            "default".to_string(),
            crate::types::model::ProfileConfig {
                adapter_path: Some("module:Adapter".to_string()),
                max_batch_tokens: Some(4096),
                compute_precision: None,
                adapter_options: None,
                extends: None,
            },
        );
        registry
            .add_model_config(ModelConfig {
                name: "test/model".to_string(),
                adapter_module: None,
                default_bundle: None,
                pool: None,
                profiles,
                inputs: None,
                max_sequence_length: Some(512),
                tasks: Some(
                    serde_yaml::from_str(
                        r#"
encode:
  dense:
    dim: 384
"#,
                    )
                    .unwrap(),
                ),
            })
            .unwrap();

        let mut profiles = HashMap::new();
        profiles.insert(
            "default".to_string(),
            crate::types::model::ProfileConfig {
                adapter_path: Some("module:Adapter".to_string()),
                max_batch_tokens: Some(4096),
                compute_precision: None,
                adapter_options: None,
                extends: None,
            },
        );
        profiles.insert(
            "alt".to_string(),
            crate::types::model::ProfileConfig {
                adapter_path: Some("module:Adapter".to_string()),
                max_batch_tokens: Some(2048),
                compute_precision: None,
                adapter_options: None,
                extends: None,
            },
        );

        registry
            .add_model_config(ModelConfig {
                name: "test/model".to_string(),
                adapter_module: None,
                default_bundle: None,
                pool: None,
                profiles,
                inputs: None,
                max_sequence_length: None,
                tasks: None,
            })
            .unwrap();

        let refreshed = registry.get_model_info("test/model").unwrap();
        assert!(refreshed.profile_names.contains("alt"));
        assert_eq!(refreshed.info_extras.outputs, vec!["dense"]);
        assert_eq!(refreshed.info_extras.dims["dense"], 384);
        assert_eq!(refreshed.info_extras.max_sequence_length, Some(512));
    }

    /// Helper for the authoritative-override and append-only-conflict tests:
    /// returns a `ModelConfig` for `test/model` with a single `default`
    /// profile pointing at `adapter_path` and a fixed `max_batch_tokens`.
    fn cfg_for_test_model(adapter_path: &str, max_batch_tokens: u32) -> ModelConfig {
        ModelConfig {
            name: "test/model".to_string(),
            adapter_module: None,
            default_bundle: None,
            pool: None,
            profiles: {
                let mut m = HashMap::new();
                m.insert(
                    "default".to_string(),
                    crate::types::model::ProfileConfig {
                        adapter_path: Some(adapter_path.to_string()),
                        max_batch_tokens: Some(max_batch_tokens),
                        compute_precision: None,
                        adapter_options: None,
                        extends: None,
                    },
                );
                m
            },
            inputs: None,
            max_sequence_length: None,
            tasks: None,
        }
    }

    /// The append-only `add_model_config` path is what the NATS pub/sub
    /// delta consumer uses. Single-epoch duplicate publishes must still be
    /// rejected, otherwise a malformed re-publish could silently overwrite
    /// the registered config. Locks the existing behavior in place so
    /// future changes do not regress it.
    #[test]
    fn test_add_model_config_append_only_rejects_conflict() {
        let (_dir, bundles_dir, models_dir) = create_test_dirs();
        fs::write(
            bundles_dir.join("default.yaml"),
            r#"
name: default
priority: 10
adapters:
  - sie_server.adapters.sentence_transformer
  - sie_server.adapters.pytorch_embedding
"#,
        )
        .unwrap();
        let registry = ModelRegistry::new(&bundles_dir, &models_dir, true);

        registry
            .add_model_config(cfg_for_test_model(
                "sie_server.adapters.sentence_transformer:SentenceTransformerAdapter",
                4096,
            ))
            .expect("first apply seeds the registry");

        // Same sie_id and profile, different adapter_path: conflict.
        let err = registry
            .add_model_config(cfg_for_test_model(
                "sie_server.adapters.pytorch_embedding:PyTorchEmbeddingAdapter",
                4096,
            ))
            .expect_err("plain add_model_config must reject conflicting profile config");
        assert!(
            err.contains("already exists with different config"),
            "unexpected error message: {err}"
        );
        assert!(
            err.contains("append-only"),
            "error must reference the append-only invariant: {err}"
        );
    }

    /// The test-only authoritative merge path preserves the old per-model
    /// conflict override behavior. Production export replay now uses
    /// `replace_model_configs_authoritative`, but this still guards the shared
    /// merge code for divergent stored profiles.
    #[test]
    fn test_add_model_config_authoritative_overrides_conflicting_profile() {
        let (_dir, bundles_dir, models_dir) = create_test_dirs();
        fs::write(
            bundles_dir.join("default.yaml"),
            r#"
name: default
priority: 10
adapters:
  - sie_server.adapters.sentence_transformer
  - sie_server.adapters.pytorch_embedding
"#,
        )
        .unwrap();
        let registry = ModelRegistry::new(&bundles_dir, &models_dir, true);

        // Seed: gateway pod cold-started against an old sie-config rev.
        registry
            .add_model_config(cfg_for_test_model(
                "sie_server.adapters.sentence_transformer:SentenceTransformerAdapter",
                4096,
            ))
            .expect("seed apply");

        // sie-config rolls forward; export now carries a new adapter for
        // the same profile. Authoritative apply must accept it.
        registry
            .add_model_config_authoritative(cfg_for_test_model(
                "sie_server.adapters.pytorch_embedding:PyTorchEmbeddingAdapter",
                8192,
            ))
            .expect("authoritative apply must override on diff, not fail");

        // Confirm the registry now holds the new config: a follow-up plain
        // apply of the same (new) config is a no-op (returns Ok with no
        // newly created profiles), proving the stored CanonicalProfile is
        // the overridden one. If the override had not taken effect, this
        // call would hit append-only and return Err.
        let (created, _skipped, _bundles) = registry
            .add_model_config(cfg_for_test_model(
                "sie_server.adapters.pytorch_embedding:PyTorchEmbeddingAdapter",
                8192,
            ))
            .expect("replay of the now-stored config must succeed under append-only");
        assert!(
            created.is_empty(),
            "replay of stored config must not report newly created profiles, got: {created:?}"
        );
    }

    /// Regression: a delta apply whose new profile ``extends`` a profile that
    /// only lives in ``existing.profile_configs`` (not in the incoming config)
    /// must still resolve. Pre-fix this hit "Profile 'a100-40gb' has a missing
    /// parent or an inheritance cycle" because resolution only walked
    /// ``config.profiles``.
    #[test]
    fn test_add_model_config_delta_resolves_extends_from_existing_profiles() {
        let (_dir, bundles_dir, models_dir) = create_test_dirs();
        fs::write(
            bundles_dir.join("default.yaml"),
            r#"
name: default
priority: 10
adapters:
  - sie_server.adapters.sentence_transformer
"#,
        )
        .unwrap();
        let registry = ModelRegistry::new(&bundles_dir, &models_dir, true);

        // Seed: one config carrying the ``default`` profile.
        registry
            .add_model_config(cfg_for_test_model(
                "sie_server.adapters.sentence_transformer:SentenceTransformerAdapter",
                4096,
            ))
            .expect("seed apply with default profile");

        // Delta: a second config bringing only ``a100-40gb`` which extends
        // ``default``. ``default`` lives only in ``existing.profile_configs``
        // at this point.
        let delta = ModelConfig {
            name: "test/model".to_string(),
            adapter_module: None,
            default_bundle: None,
            pool: None,
            profiles: {
                let mut m = HashMap::new();
                m.insert(
                    "a100-40gb".to_string(),
                    crate::types::model::ProfileConfig {
                        adapter_path: None,
                        max_batch_tokens: Some(8192),
                        compute_precision: None,
                        adapter_options: None,
                        extends: Some("default".to_string()),
                    },
                );
                m
            },
            inputs: None,
            max_sequence_length: None,
            tasks: None,
        };

        let (created, _skipped, _bundles) = registry
            .add_model_config(delta.clone())
            .expect("delta apply must resolve extends against existing.profile_configs");
        assert_eq!(created, vec!["a100-40gb".to_string()]);

        // Same delta under the test-only authoritative path must also succeed
        // because the override path uses the same merged resolution map.
        registry
            .add_model_config_authoritative(delta)
            .expect("authoritative replay of the same delta must also resolve cleanly");
    }

    /// M10 follow-up: a delta-update that adds a profile with new
    /// ``adapter_options.loadtime.lora_paths`` must refresh both the
    /// union (`info_extras.lora_adapters`) and the per-profile map
    /// (`info_extras.profile_lora_adapters`). The previous merge path
    /// only refreshed inputs/outputs/dims/max_*; LoRA capabilities
    /// stayed stale until a full reload, so the gateway's capability
    /// gate could reject a request for an adapter the worker had
    /// actually loaded after the delta.
    #[test]
    fn test_add_model_config_delta_refreshes_lora_capabilities() {
        let (_dir, bundles_dir, models_dir) = create_test_dirs();
        fs::write(
            bundles_dir.join("default.yaml"),
            r#"
name: default
priority: 10
adapters:
  - sie_server.adapters.sentence_transformer
"#,
        )
        .unwrap();
        let registry = ModelRegistry::new(&bundles_dir, &models_dir, true);

        // Seed: ``default`` profile with one LoRA adapter ``base-a``.
        let seed = ModelConfig {
            name: "test/model".to_string(),
            adapter_module: None,
            default_bundle: None,
            pool: None,
            profiles: {
                let mut m = HashMap::new();
                m.insert(
                    "default".to_string(),
                    crate::types::model::ProfileConfig {
                        adapter_path: Some(
                            "sie_server.adapters.sentence_transformer:SentenceTransformerAdapter"
                                .to_string(),
                        ),
                        max_batch_tokens: Some(4096),
                        compute_precision: None,
                        adapter_options: Some(serde_json::json!({
                            "loadtime": {
                                "lora_paths": { "base-a": "hf://repo/base-a" },
                            },
                        })),
                        extends: None,
                    },
                );
                m
            },
            inputs: None,
            max_sequence_length: None,
            tasks: None,
        };
        registry.add_model_config(seed).expect("seed");

        // Sanity: union and per-profile map both carry ``base-a`` only.
        let snap = registry.snapshot.load();
        let entry = snap
            .models
            .get("test/model")
            .expect("seed produced an entry");
        assert_eq!(
            entry.info_extras.lora_adapters.as_deref().unwrap_or(&[]),
            &["base-a".to_string()],
        );
        let per_profile_seed = entry
            .info_extras
            .profile_lora_adapters
            .as_ref()
            .expect("seed populated profile_lora_adapters");
        assert_eq!(
            per_profile_seed.get("default").map(Vec::as_slice),
            Some(&["base-a".to_string()][..]),
        );

        // Delta: add ``a100-40gb`` profile with a *different* LoRA
        // adapter ``a100-adapter``. The merge must surface both
        // adapters in the union and scope each to its own profile in
        // the per-profile map. Previously this stayed stale until a
        // full reload.
        let delta = ModelConfig {
            name: "test/model".to_string(),
            adapter_module: None,
            default_bundle: None,
            pool: None,
            profiles: {
                let mut m = HashMap::new();
                m.insert(
                    "a100-40gb".to_string(),
                    crate::types::model::ProfileConfig {
                        adapter_path: Some(
                            "sie_server.adapters.sentence_transformer:SentenceTransformerAdapter"
                                .to_string(),
                        ),
                        max_batch_tokens: Some(8192),
                        compute_precision: None,
                        adapter_options: Some(serde_json::json!({
                            "loadtime": {
                                "lora_paths": { "a100-adapter": "hf://repo/a100" },
                            },
                        })),
                        extends: None,
                    },
                );
                m
            },
            inputs: None,
            max_sequence_length: None,
            tasks: None,
        };
        registry
            .add_model_config(delta)
            .expect("delta adds new profile + lora");

        let snap = registry.snapshot.load();
        let entry = snap.models.get("test/model").expect("entry survived");
        let union: Vec<String> = entry
            .info_extras
            .lora_adapters
            .clone()
            .unwrap_or_default()
            .into_iter()
            .collect();
        // Union now lists both adapters (order is insertion-defined; sort to
        // compare).
        let mut union_sorted = union;
        union_sorted.sort();
        assert_eq!(
            union_sorted,
            vec!["a100-adapter".to_string(), "base-a".to_string()],
            "delta-update did not refresh the union of lora_adapters",
        );
        let per_profile = entry
            .info_extras
            .profile_lora_adapters
            .as_ref()
            .expect("profile_lora_adapters present");
        assert_eq!(
            per_profile.get("default").map(Vec::as_slice),
            Some(&["base-a".to_string()][..]),
            "delta-update wiped the existing default profile's adapter list",
        );
        assert_eq!(
            per_profile.get("a100-40gb").map(Vec::as_slice),
            Some(&["a100-adapter".to_string()][..]),
            "delta-update did not add the new profile's adapter",
        );
    }

    #[test]
    fn test_add_model_config_authoritative_rebuilds_adapter_modules() {
        let (_dir, bundles_dir, models_dir) = create_test_dirs();
        fs::write(
            bundles_dir.join("old.yaml"),
            r#"
name: old
priority: 10
adapters:
  - old.adapter
"#,
        )
        .unwrap();
        fs::write(
            bundles_dir.join("new.yaml"),
            r#"
name: new
priority: 10
adapters:
  - new.adapter
"#,
        )
        .unwrap();
        let registry = ModelRegistry::new(&bundles_dir, &models_dir, true);

        registry
            .add_model_config(cfg_for_test_model("old.adapter:Adapter", 4096))
            .expect("seed apply");
        assert_eq!(registry.resolve_bundle("test/model", None).unwrap(), "old");

        registry
            .add_model_config_authoritative(cfg_for_test_model("new.adapter:Adapter", 4096))
            .expect("authoritative apply must move adapter module");

        let entry = registry.get_model_info("test/model").unwrap();
        assert_eq!(
            entry.adapter_modules,
            HashSet::from(["new.adapter".to_string()])
        );
        assert_eq!(registry.resolve_bundle("test/model", None).unwrap(), "new");
    }

    #[test]
    fn test_compute_bundle_config_hash() {
        let (_dir, bundles_dir, models_dir) = create_test_dirs();

        fs::write(
            bundles_dir.join("default.yaml"),
            r#"
name: default
priority: 10
adapters:
  - sie_server.adapters.sentence_transformer
"#,
        )
        .unwrap();

        fs::write(
            models_dir.join("bge-m3.yaml"),
            r#"
name: BAAI/bge-m3
profiles:
  default:
    adapter_path: "sie_server.adapters.sentence_transformer:SentenceTransformerAdapter"
    max_batch_tokens: 4096
"#,
        )
        .unwrap();

        let registry = ModelRegistry::new(&bundles_dir, &models_dir, true);
        let hash = registry.compute_bundle_config_hash("default");
        assert!(!hash.is_empty());
        assert_eq!(hash.len(), 64); // SHA-256 hex

        // Hash should be deterministic
        let hash2 = registry.compute_bundle_config_hash("default");
        assert_eq!(hash, hash2);

        // Unknown bundle returns empty
        let empty = registry.compute_bundle_config_hash("nonexistent");
        assert!(empty.is_empty());
    }

    #[test]
    fn test_compute_bundle_config_hash_for_pool_filters_model_pool() {
        let (_dir, bundles_dir, models_dir) = create_test_dirs();

        fs::write(
            bundles_dir.join("default.yaml"),
            r#"
name: default
priority: 10
adapters:
  - sie_server.adapters.sentence_transformer
"#,
        )
        .unwrap();

        fs::write(
            models_dir.join("default.yaml"),
            r#"
name: default/model
profiles:
  default:
    adapter_path: "sie_server.adapters.sentence_transformer:SentenceTransformerAdapter"
    max_batch_tokens: 4096
"#,
        )
        .unwrap();

        fs::write(
            models_dir.join("tenant.yaml"),
            r#"
name: tenant/model
pool: customer-a
profiles:
  default:
    adapter_path: "sie_server.adapters.sentence_transformer:SentenceTransformerAdapter"
    max_batch_tokens: 4096
"#,
        )
        .unwrap();

        let registry = ModelRegistry::new(&bundles_dir, &models_dir, true);
        let global_hash = registry.compute_bundle_config_hash("default");
        let default_hash = registry.compute_bundle_config_hash_for_pool("default", "default");
        let tenant_hash = registry.compute_bundle_config_hash_for_pool("default", "customer-a");

        assert_eq!(
            registry.get_model_pool_name("tenant/model").as_deref(),
            Some("customer-a")
        );
        assert!(!global_hash.is_empty());
        assert!(!default_hash.is_empty());
        assert!(!tenant_hash.is_empty());
        assert_ne!(global_hash, default_hash);
        assert_ne!(global_hash, tenant_hash);
        assert_ne!(default_hash, tenant_hash);
        assert_eq!(
            registry.compute_bundle_config_hash_for_pool("default", "missing"),
            ""
        );

        registry.install_bundle_config_hashes(HashMap::from([(
            "default".to_string(),
            "control-plane-global-hash".to_string(),
        )]));
        assert_eq!(
            registry.compute_bundle_config_hash("default"),
            "control-plane-global-hash"
        );
        assert_eq!(
            registry.compute_bundle_config_hash_for_pool("default", "default"),
            default_hash
        );
        assert_eq!(
            registry.compute_bundle_config_hash_for_pool("default", "customer-a"),
            tenant_hash
        );
    }

    #[test]
    fn test_compute_bundle_config_hash_ignores_profile_variant_aliases() {
        let (_dir, bundles_dir, models_dir) = create_test_dirs();

        fs::write(
            bundles_dir.join("default.yaml"),
            r#"
name: default
priority: 10
adapters:
  - sie_server.adapters.sentence_transformer
"#,
        )
        .unwrap();

        fs::write(
            models_dir.join("example.yaml"),
            r#"
name: example/model
profiles:
  default:
    adapter_path: "sie_server.adapters.sentence_transformer:SentenceTransformerAdapter"
    max_batch_tokens: 4096
  small:
    adapter_path: "sie_server.adapters.sentence_transformer:SentenceTransformerAdapter"
    max_batch_tokens: 2048
    compute_precision: fp16
"#,
        )
        .unwrap();

        let registry = ModelRegistry::new(&bundles_dir, &models_dir, true);
        assert!(registry.model_exists("example/model:small"));

        assert_eq!(
            registry.compute_bundle_config_hash("default"),
            "a4ce038148c529799f88014dde9360175c9e9536f353162cf3bc312c09f4cd23"
        );
    }

    #[test]
    fn test_install_bundle_config_hashes_overrides_local_hash() {
        let (_dir, bundles_dir, models_dir) = create_test_dirs();

        fs::write(
            bundles_dir.join("default.yaml"),
            r#"
name: default
priority: 10
adapters:
  - sie_server.adapters.sentence_transformer
"#,
        )
        .unwrap();

        fs::write(
            models_dir.join("bge-m3.yaml"),
            r#"
name: BAAI/bge-m3
profiles:
  default:
    adapter_path: "sie_server.adapters.sentence_transformer:SentenceTransformerAdapter"
    max_batch_tokens: 4096
"#,
        )
        .unwrap();

        let registry = ModelRegistry::new(&bundles_dir, &models_dir, true);
        assert_ne!(
            registry.compute_bundle_config_hash("default"),
            "authority-hash"
        );

        registry.install_bundle_config_hashes(HashMap::from([(
            "default".to_string(),
            "authority-hash".to_string(),
        )]));
        assert_eq!(
            registry.compute_bundle_config_hash("default"),
            "authority-hash"
        );

        registry
            .install_bundle_config_hashes(HashMap::from([("default".to_string(), String::new())]));
        assert_eq!(registry.compute_bundle_config_hash("default"), "");
    }

    #[test]
    fn test_concurrent_add_model_config_preserves_all_writes() {
        // Fix #4 regression: without `write_lock`, two threads that call
        // `add_model_config` in parallel can both `snapshot.load()` the
        // same base, both build a derived clone with only their own
        // model, and the second `store()` silently clobbers the first.
        // This test fires 32 threads at the same registry and asserts
        // every model ended up visible.
        use std::sync::Arc;
        use std::thread;

        let (_dir, bundles_dir, models_dir) = create_test_dirs();
        fs::write(
            bundles_dir.join("default.yaml"),
            r#"
name: default
priority: 10
adapters:
  - sie_server.adapters.sentence_transformer
"#,
        )
        .unwrap();
        let registry = Arc::new(ModelRegistry::new(&bundles_dir, &models_dir, true));

        let n: usize = 32;
        let mut handles = Vec::with_capacity(n);
        for i in 0..n {
            let reg = Arc::clone(&registry);
            handles.push(thread::spawn(move || {
                let config = ModelConfig {
                    name: format!("race/model-{i}"),
                    adapter_module: None,
                    default_bundle: None,
                    pool: None,
                    profiles: {
                        let mut m = HashMap::new();
                        m.insert(
                            "default".to_string(),
                            crate::types::model::ProfileConfig {
                                adapter_path: Some(
                                    "sie_server.adapters.sentence_transformer:A".to_string(),
                                ),
                                max_batch_tokens: Some(4096),
                                compute_precision: None,
                                adapter_options: None,
                                extends: None,
                            },
                        );
                        m
                    },
                    inputs: None,
                    max_sequence_length: None,
                    tasks: None,
                };
                reg.add_model_config(config).expect("add_model_config");
            }));
        }
        for h in handles {
            h.join().expect("thread join");
        }

        for i in 0..n {
            let id = format!("race/model-{i}");
            assert!(
                registry.model_exists(&id),
                "{id} should be present after concurrent adds; if this fails \
                 the ArcSwap load-clone-store is racing (bug #4 regression)."
            );
        }
    }

    #[test]
    fn test_reload() {
        let (_dir, bundles_dir, models_dir) = create_test_dirs();

        let registry = ModelRegistry::new(&bundles_dir, &models_dir, true);
        assert!(registry.list_models().is_empty());

        // Add files and reload
        fs::write(
            bundles_dir.join("default.yaml"),
            r#"
name: default
priority: 10
adapters:
  - sie_server.adapters.sentence_transformer
"#,
        )
        .unwrap();

        fs::write(
            models_dir.join("model.yaml"),
            r#"
name: test/model
profiles:
  default:
    adapter_path: "sie_server.adapters.sentence_transformer:Adapter"
"#,
        )
        .unwrap();

        registry.reload();
        assert_eq!(registry.list_models().len(), 1);
        assert_eq!(registry.list_bundles().len(), 1);
    }

    // ---------------------------------------------------------------
    // engine field. These tests pin the wire shape and validation
    // contract so a future refactor can't quietly drop the field.
    // Today only ``"pytorch"`` is recognised — see ``KNOWN_ENGINES``.
    // ---------------------------------------------------------------

    #[test]
    fn test_engine_defaults_to_pytorch_when_omitted() {
        let (_dir, bundles_dir, models_dir) = create_test_dirs();
        fs::write(
            bundles_dir.join("default.yaml"),
            "name: default\npriority: 10\nadapters:\n  - sie_server.adapters.sentence_transformer\n",
        )
        .unwrap();

        let registry = ModelRegistry::new(&bundles_dir, &models_dir, true);
        let info = registry.get_bundle_info("default").unwrap();
        assert_eq!(info.engine, DEFAULT_ENGINE);
        assert_eq!(info.engine, "pytorch");
    }

    #[test]
    fn test_engine_unknown_value_skips_bundle() {
        // Unknown engine values are a hard error — silently coercing
        // would let a typo'd ``engine: pytroch`` mis-route traffic.
        // We surface this as "bundle skipped";
        // the registry still loads, but the bad bundle is absent.
        let (_dir, bundles_dir, models_dir) = create_test_dirs();
        fs::write(
            bundles_dir.join("good.yaml"),
            "name: good\npriority: 10\nadapters:\n  - sie_server.adapters.sentence_transformer\n",
        )
        .unwrap();
        fs::write(
            bundles_dir.join("typo.yaml"),
            "name: typo\nengine: pytroch\npriority: 10\nadapters:\n  - sie_server.adapters.foo\n",
        )
        .unwrap();

        let registry = ModelRegistry::new(&bundles_dir, &models_dir, true);
        let bundles = registry.list_bundles();
        assert!(bundles.contains(&"good".to_string()));
        assert!(
            !bundles.contains(&"typo".to_string()),
            "bundle with unknown engine must be rejected at load time, not silently coerced"
        );
    }

    #[test]
    fn test_resolve_bundle_with_engine_filter_passes_through_for_only_engine() {
        // Today there's exactly one engine in ``KNOWN_ENGINES``
        // (``"pytorch"``), so the engine-filter codepath must be a
        // no-op for any pytorch-backed bundle: with or without the
        // filter, the same bundle resolves. Re-introduce a
        // multi-engine version of this test when a second engine
        // lands.
        let (_dir, bundles_dir, models_dir) = create_test_dirs();
        fs::write(
            bundles_dir.join("default.yaml"),
            r#"
name: default
priority: 10
adapters:
  - sie_server.adapters.bert_flash
"#,
        )
        .unwrap();
        fs::write(
            models_dir.join("bert.yaml"),
            r#"
name: example/bert
profiles:
  default:
    adapter_path: "sie_server.adapters.bert_flash:Adapter"
"#,
        )
        .unwrap();

        let registry = ModelRegistry::new(&bundles_dir, &models_dir, true);

        let plain = registry.resolve_bundle("example/bert", None).unwrap();
        assert_eq!(plain, "default");

        let filtered = registry
            .resolve_bundle_with_engine("example/bert", None, Some("pytorch"))
            .unwrap();
        assert_eq!(filtered, "default");
    }

    #[test]
    fn test_engine_known_engines_lockstep_with_bundle_module() {
        // Belt-and-braces: prove the registry's KNOWN_ENGINES import
        // resolves to the exact constant exported by the bundle types
        // module, so a future move/rename trips compile-time rather
        // than runtime. Also enforces the invariant that every known
        // engine has a non-empty adapter-prefix list — without it the
        // namespace check is a no-op.
        for engine in KNOWN_ENGINES {
            assert!(
                !engine_adapter_prefixes(engine).is_empty(),
                "every engine in KNOWN_ENGINES must have a non-empty \
                 adapter-prefix list (engine={engine}); add it to \
                 engine_adapter_prefixes() in types/bundle.rs"
            );
        }
    }
}
