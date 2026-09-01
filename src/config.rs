// SPDX-License-Identifier: GPL-3.0-or-later OR AGPL-3.0-or-later
// Copyright (C) 2025-2026  Red Hat, Inc.

use crate::gcp_auth;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    endpoints: Vec<EndpointConfig>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EndpointConfig {
    pub name: String,
    pub url: String,
    #[serde(default = "default_timeout")]
    pub timeout: u64,
    #[serde(default = "default_retries")]
    pub retries: u32,
    #[serde(default = "default_delay")]
    pub delay: u64,
    #[serde(default = "default_max_delay")]
    pub max_delay: u64,
    #[serde(default)]
    pub wait: u64,
    pub root_certificate_pem: Option<String>,
    pub api_key_file: Box<Option<String>>,
    pub x_api_key_file: Box<Option<String>>,
    pub gcp_service_account_file: Box<Option<String>>,
    #[serde(skip)]
    pub gcp_token_provider: Option<std::sync::Arc<gcp_auth::GcpTokenProvider>>,
    pub context: Option<EndpointContext>,
    pub json: Option<EndpointJson>,
    pub headers: Option<EndpointHeaders>,
    #[serde(default)]
    pub primary: bool,
    #[serde(default = "default_use_backticks")]
    pub use_backticks: bool,
    #[serde(flatten)]
    pub config: EndpointTypeConfig,
}

fn default_timeout() -> u64 {
    3600000
}

fn default_retries() -> u32 {
    100
}

fn default_delay() -> u64 {
    10000
}

fn default_max_delay() -> u64 {
    600000
}

fn default_use_backticks() -> bool {
    true
}

macro_rules! check_conflicting_context_fields {
    ($endpoint_context:expr, $variant_context:expr, $variant_name:expr, $endpoint_index:expr, $path:expr, $j:expr, $($field:ident),*) => {
        $(
            if $endpoint_context.$field.is_some() && $variant_context.$field.is_some() {
                return Err(anyhow::anyhow!(
                    "Endpoint {} in config file {} has conflicting context configuration in variant {} at index {}: {} is defined in both endpoint and variant",
                    $endpoint_index,
                    $path.display(),
                    $variant_name,
                    $j,
                    stringify!($field)
                ));
            }
        )*
    };
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum EndpointTypeConfig {
    #[serde(rename = "openai")]
    OpenAI {
        variants: Option<Vec<EndpointVariants>>,
        #[serde(default)]
        no_chat: bool, // false: /v1/chat/completions true /v1/completions
        #[serde(default)]
        gbnf: bool,
    },
    #[serde(rename = "anthropic")]
    Anthropic {
        variants: Option<Vec<EndpointVariants>>,
    },
    #[serde(rename = "patchpal")]
    Patchpal {
        #[serde(default)]
        telemetry: bool,
        #[serde(default = "default_n_beams")]
        n_beams: u32,
    },
}

fn default_n_beams() -> u32 {
    3
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct EndpointVariants {
    pub name: Box<Option<String>>,
    pub context: Option<EndpointContext>,
    pub json: Option<EndpointJson>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EndpointContext {
    #[serde(default)]
    pub layout: Option<EndpointContextLayout>,
    #[serde(default)]
    pub no_diff: Option<bool>,
    #[serde(default)]
    pub no_training: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EndpointContextLayout {
    #[serde(default)]
    pub system_message: Vec<EndpointContextElement>,
    #[serde(default)]
    pub user_message: Vec<EndpointContextElement>,
}

impl Default for EndpointContextLayout {
    fn default() -> Self {
        EndpointContextLayout {
            system_message: vec![
                EndpointContextElement::Training,
                EndpointContextElement::Diff,
                EndpointContextElement::Prompt,
            ],
            user_message: Vec::new(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub enum EndpointContextElement {
    Prompt,
    Training,
    Diff,
}

// impl EndpointContextElement {
//     fn to_string(&self) -> &'static str {
//         match self {
//             EndpointContextElement::Prompt => "prompt",
//             EndpointContextElement::Training => "training",
//             EndpointContextElement::Diff => "diff",
//         }
//     }
//}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EndpointJson {
    #[serde(flatten)]
    pub json: std::collections::HashMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EndpointHeaders {
    #[serde(flatten)]
    pub headers: std::collections::HashMap<String, serde_json::Value>,
}

impl Config {
    const FORBIDDEN_CHARS: &str = "()|,#$[]";

    pub fn load(path: &Path) -> Result<Self> {
        let contents = fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {}", path.display()))?;

        let mut config: Config = serde_yaml::from_str(&contents)
            .with_context(|| format!("Failed to parse config file as YAML: {}", path.display()))?;

        if config.endpoints.is_empty() {
            return Err(anyhow::anyhow!(
                "No endpoints configured in config file: {}",
                path.display()
            ));
        }

        // Trim whitespace from endpoint
        Self::trim_endpoint_whitespace(&mut config.endpoints);

        // Check that each endpoint has required fields
        for (i, endpoint) in config.endpoints.iter().enumerate() {
            if endpoint.name.is_empty() {
                return Err(anyhow::anyhow!(
                    "Endpoint {} in config file {} has empty name",
                    i,
                    path.display()
                ));
            }
            if endpoint
                .name
                .chars()
                .any(|c| Self::FORBIDDEN_CHARS.contains(c))
            {
                return Err(anyhow::anyhow!(
                    "Endpoint {} in config file {} has invalid name '{}' contains {} chars",
                    i,
                    path.display(),
                    endpoint.name,
                    Self::FORBIDDEN_CHARS
                ));
            }
            if endpoint.url.is_empty() {
                return Err(anyhow::anyhow!(
                    "Endpoint {} in config file {} has empty url",
                    i,
                    path.display()
                ));
            }

            // Validate OpenAI endpoint configuration
            Self::validate_endpoint(endpoint, i, path)?;
        }

        let mut seen_names = std::collections::HashSet::new();
        for (i, endpoint) in config.endpoints.iter().enumerate() {
            if !seen_names.insert(&endpoint.name) {
                return Err(anyhow::anyhow!(
                    "Endpoint {} in config file {} has duplicate name '{}'",
                    i,
                    path.display(),
                    endpoint.name
                ));
            }
        }

        Self::validate_primary(&mut config.endpoints);

        Self::initialize_gcp_token_providers(&mut config.endpoints, path)?;

        log::debug!("{:?}", config);

        Ok(config)
    }

    fn validate_primary(endpoints: &mut [EndpointConfig]) {
        if !endpoints.iter().any(|e| e.primary) {
            endpoints.iter_mut().for_each(|e| e.primary = true);
        }
    }

    fn initialize_gcp_token_providers(endpoints: &mut [EndpointConfig], path: &Path) -> Result<()> {
        for (i, endpoint) in endpoints.iter_mut().enumerate() {
            if endpoint.api_key_file.as_deref().is_some()
                && endpoint.gcp_service_account_file.as_deref().is_some()
            {
                return Err(anyhow::anyhow!(
                    "Endpoint {} in config file {} has both api_key_file and gcp_service_account_file set",
                    i,
                    path.display()
                ));
            }
            if let Some(gcp_service_account_file) = &*endpoint.gcp_service_account_file {
                endpoint.gcp_token_provider = Some(std::sync::Arc::new(
                    gcp_auth::GcpTokenProvider::from_file(gcp_service_account_file.clone()),
                ));
            }
        }
        Ok(())
    }

    fn trim_endpoint_whitespace(endpoints: &mut [EndpointConfig]) {
        for endpoint in endpoints {
            endpoint.name = endpoint.name.trim().to_string();
            endpoint.url = endpoint.url.trim().to_string();
            Self::trim_variants_whitespace(&mut endpoint.config);
        }
    }

    fn trim_variants_whitespace(config: &mut EndpointTypeConfig) {
        if let EndpointTypeConfig::OpenAI { variants, .. }
        | EndpointTypeConfig::Anthropic { variants, .. } = config
            && let Some(variant_list) = variants
        {
            for variant in variant_list {
                if let Some(variant) = &mut *variant.name {
                    *variant = variant.trim().to_string();
                }
            }
        }
    }

    fn validate_endpoint(endpoint: &EndpointConfig, index: usize, path: &Path) -> Result<()> {
        if let EndpointTypeConfig::OpenAI { variants, .. }
        | EndpointTypeConfig::Anthropic { variants, .. } = &endpoint.config
        {
            // Check that all variant names are unique
            Self::validate_variants(variants, index, path, &endpoint.json, &endpoint.context)?;
        }
        Ok(())
    }

    fn validate_variants(
        variants_list: &Option<Vec<EndpointVariants>>,
        endpoint_index: usize,
        path: &Path,
        endpoint_json: &Option<EndpointJson>,
        endpoint_context: &Option<EndpointContext>,
    ) -> Result<()> {
        if let Some(variants_list) = variants_list {
            // Check that all variant names are unique
            let mut seen_variants = std::collections::HashSet::new();
            // Collect all keys from endpoint.json
            let mut seen_keys = std::collections::HashSet::new();
            if let Some(endpoint_json) = endpoint_json {
                for key in endpoint_json.json.keys() {
                    if !seen_keys.insert(key) {
                        return Err(anyhow::anyhow!(
                            "Endpoint {} in config file {} has duplicate key '{}' in endpoint.json",
                            endpoint_index,
                            path.display(),
                            key
                        ));
                    }
                }
            }
            for (j, variant) in variants_list.iter().enumerate() {
                let variant_name = if let Some(v) = &*variant.name {
                    v.to_string()
                } else {
                    "".to_string()
                };
                if variant_name
                    .chars()
                    .any(|c| Self::FORBIDDEN_CHARS.contains(c))
                {
                    return Err(anyhow::anyhow!(
                        "Endpoint {} in config file {} has invalid variant name '{}' at index {} contains {} chars",
                        endpoint_index,
                        path.display(),
                        variant_name,
                        j,
                        Self::FORBIDDEN_CHARS
                    ));
                }
                if !seen_variants.insert(variant_name) {
                    return Err(anyhow::anyhow!(
                        "Endpoint {} in config file {} has duplicate variant name '{}' at index {}",
                        endpoint_index,
                        path.display(),
                        variant.name.as_deref().unwrap_or(""),
                        j
                    ));
                }
                // Check for duplicate keys between endpoint.json and variant.json
                if let Some(variant_json) = &variant.json {
                    for key in variant_json.json.keys() {
                        if seen_keys.contains(key) {
                            return Err(anyhow::anyhow!(
                                "Endpoint {} in config file {} has duplicate key '{}' in variant {} at index {}",
                                endpoint_index,
                                path.display(),
                                key,
                                variant.name.clone().unwrap_or("\"\"".to_string()),
                                j
                            ));
                        }
                    }
                }
                // Check if endpoint_context.is_some() and variant.context.is_some(),
                // then a field can be Some only in either the endpoint.context or the
                // variant.context but not in both
                if let (Some(endpoint_context), Some(variant_context)) =
                    (&endpoint_context, &variant.context)
                {
                    check_conflicting_context_fields!(
                        endpoint_context,
                        variant_context,
                        variant.name.clone().unwrap_or("\"\"".to_string()),
                        endpoint_index,
                        path,
                        j,
                        layout,
                        no_diff,
                        no_training
                    );
                }
            }
        }
        Ok(())
    }

    pub fn get_all_endpoints(&self) -> &[EndpointConfig] {
        &self.endpoints
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_loading() {
        let config_yaml = include_str!(concat!("../", env!("CARGO_PKG_NAME"), ".yaml"));
        let config: Config = serde_yaml::from_str(config_yaml).unwrap();
        assert_eq!(config.endpoints.len(), 13);
        assert_eq!(config.endpoints[0].name, "Gemini 3.7 Flash");
        assert_eq!(config.endpoints[1].name, "Gemini 3.1 Pro");
        assert_eq!(config.endpoints[2].name, "gpt-5.6-sol");
        assert_eq!(config.endpoints[3].name, "gpt-5.5");
        assert_eq!(config.endpoints[4].name, "Claude Opus 5");
        assert_eq!(config.endpoints[5].name, "Vertex Claude Opus 5");
        assert_eq!(config.endpoints[6].name, "Claude Sonnet 5");
        assert_eq!(config.endpoints[7].name, "Vertex Claude Sonnet 5");
        assert_eq!(config.endpoints[8].name, "Vertex Gateway Claude Sonnet 4.0");
        assert_eq!(config.endpoints[9].name, "Patchpal AI");
        assert_eq!(config.endpoints[10].name, "llama.cpp vulkan minimal");
        assert_eq!(config.endpoints[11].name, "llama.cpp vulkan");
        assert_eq!(config.endpoints[12].name, "llama.cpp vulkan no_chat");
    }
}

// Local Variables:
// rust-format-on-save: t
// End:
