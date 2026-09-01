// SPDX-License-Identifier: GPL-3.0-or-later OR AGPL-3.0-or-later
// Copyright (C) 2025  Red Hat, Inc.

use crate::conflict_resolver::ConflictResolver;
use serde_json::Value;

/// Calculate the response logprob from the token logprobs
///
/// If no logprobs are available, returns None
pub fn logprob(json: &Value, perplexity: &mut Vec<String>) -> Option<f64> {
    // Check if logprobs exist in the response
    let logprobs = json
        .get("choices")
        .and_then(|c| c.as_array().and_then(|arr| arr.first()))
        .and_then(|c| c.get("logprobs"));

    // If no logprobs, return None
    let logprobs = logprobs?;

    // Extract content logprobs
    let content_logprobs = match logprobs.get("content") {
        Some(content) => content.as_array(),
        None => return None,
    };

    let content_logprobs = content_logprobs?;

    // If no content logprobs, return None
    if content_logprobs.is_empty() {
        return None;
    }

    // Find minimum log probability and track position of token with smallest distance in top_logprobs
    let mut min_logprob = f64::INFINITY;
    let mut min_logprob_pos: Option<usize> = None;
    let mut raw_min_logprob = f64::INFINITY;
    let mut raw_min_logprob_pos: Option<usize> = None;
    let mut perplexity_pos = Vec::new();
    let mut concatenated_tokens = String::new();

    // First, concatenate all tokens to find the positions of PATCHED_CODE_START and PATCHED_CODE_END
    let mut all_tokens = String::new();
    for token_logprob in content_logprobs.iter() {
        let token = token_logprob.get("token").and_then(|t| t.as_str())?;
        all_tokens.push_str(token);
    }

    // Find the positions of PATCHED_CODE_START and PATCHED_CODE_END
    let patched_code_start = &format!("{}\n", ConflictResolver::PATCHED_CODE_START);
    let patched_code_end = ConflictResolver::PATCHED_CODE_END;
    let start_pos = all_tokens.find(patched_code_start)? + patched_code_start.len();
    let end_pos = all_tokens.find(patched_code_end)?;

    for (i, token_logprob) in content_logprobs.iter().enumerate() {
        // Extract logprob value
        let logprob = token_logprob.get("logprob")?.as_f64()?;

        let token = token_logprob.get("token").and_then(|t| t.as_str())?;
        if i == content_logprobs.len() - 1 && token.is_empty() {
            continue;
        }

        let str_offset = concatenated_tokens.len();
        concatenated_tokens.push_str(token);

        if logprob < raw_min_logprob {
            raw_min_logprob = logprob;
            raw_min_logprob_pos = Some(i);
        }

        if str_offset < start_pos || str_offset >= end_pos {
            continue;
        }

        // Check top_logprobs for this token to find the one with smallest distance
        if let Some(top_logprobs) = token_logprob.get("top_logprobs").and_then(|t| t.as_array())
            && top_logprobs.len() == 2
        {
            let mut min_top_logprob = f64::INFINITY;
            let mut max_top_logprob = f64::NEG_INFINITY;
            for top_logprob in top_logprobs {
                let logprob = top_logprob.get("logprob").and_then(|lp| lp.as_f64())?;
                min_top_logprob = min_top_logprob.min(logprob);
                max_top_logprob = max_top_logprob.max(logprob);
            }
            let distance = max_top_logprob - min_top_logprob;
            perplexity_pos.push((distance * -max_top_logprob, i));
        }

        if logprob < min_logprob {
            min_logprob = logprob;
            min_logprob_pos = Some(i);
        }
    }

    // If no valid logprobs found, return None
    if !min_logprob.is_finite() {
        return None;
    }

    // Extract tokens from logprobs
    let tokens = logprobs
        .as_object()
        .and_then(|lp| lp.get("content"))
        .and_then(|c| c.as_array())?;

    perplexity_pos.sort_unstable_by(|a, b| f64::total_cmp(&b.0, &a.0));
    let perplexity_pos: Vec<_> = perplexity_pos.iter().map(|x| x.1).collect();
    perplexity_search(content_logprobs, tokens, &perplexity_pos, perplexity)?;

    // Call function with json and position of lowest logprob token
    print_logprob_diff(tokens, raw_min_logprob_pos, "~~~");
    if raw_min_logprob_pos != min_logprob_pos {
        print_logprob_diff(tokens, min_logprob_pos, "~=~");
    }

    Some(min_logprob)
}

fn perplexity_search(
    logprobs: &[Value],
    tokens: &[Value],
    perplexity_pos: &Vec<usize>,
    perplexity: &mut Vec<String>,
) -> Option<()> {
    const PERPLEXITY_BEAMS: usize = 3;
    for pos in perplexity_pos {
        let token = tokens.get(*pos)?;
        let text = token.get("token").and_then(|t| t.as_str())?;
        if text.chars().last().is_some_and(char::is_whitespace) || text.is_empty() {
            continue;
        }

        let mut concatenated_tokens = String::new();
        for token_logprob in logprobs.iter().take(*pos) {
            let token = token_logprob.get("token").and_then(|t| t.as_str())?;
            concatenated_tokens.push_str(token);
        }

        let top_logprobs = token.get("top_logprobs").and_then(|t| t.as_array())?;
        for top_logprob in top_logprobs {
            let top_text = top_logprob.get("token").and_then(|t| t.as_str())?;
            if top_text != text {
                if !top_text.chars().last().is_some_and(char::is_whitespace) && !top_text.is_empty()
                {
                    concatenated_tokens.push_str(top_text);
                    perplexity.push(concatenated_tokens);
                }
                break;
            }
        }
        if perplexity.len() >= PERPLEXITY_BEAMS - 1 {
            break;
        }
    }

    Some(())
}

fn print_logprob_diff(tokens: &[Value], pos: Option<usize>, separator: &str) -> Option<()> {
    if let Some(pos) = pos {
        // Extract tokens up to the position of the minimum logprob token
        let tokens_up_to_min: Vec<&Value> = tokens.iter().take(pos).collect();
        let mut concatenated_tokens = String::new();

        for token in tokens_up_to_min {
            let text = token.get("token").and_then(|t| t.as_str())?;
            concatenated_tokens.push_str(text);
        }

        let tokens_from_min: Vec<&Value> = tokens.iter().skip(pos).collect();
        let mut concatenated_rest = String::new();

        for token in tokens_from_min {
            let text = token.get("token").and_then(|t| t.as_str())?;
            concatenated_rest.push_str(text);
        }

        // Print the concatenated tokens
        log::trace!(
            "Logprob:\n{}{separator}{}",
            concatenated_tokens,
            concatenated_rest
        );
    }
    Some(())
}

pub fn logprob_to_prob(logprob: f64) -> f64 {
    //logprob.exp().min(1.0) * 100.
    1000000_f64.powf(logprob).clamp(0., 1.) * 100.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logprob_with_logprobs() {
        let json_str = &format!(
            r#"{{
            "choices": [
                {{
                    "logprobs": {{
                        "content": [
                            {{
                                "logprob": 0.0,
				"token": "{}\n"
                            }},
                            {{
                                "logprob": -1.0,
				"token": "2"
                            }},
                            {{
                                "logprob": -2.0,
				"token": " "
                            }},
                            {{
                                "logprob": -3.0,
				"token": "{}"
                            }}
                        ]
                    }}
                }}
            ]
        }}"#,
            &ConflictResolver::PATCHED_CODE_START,
            &ConflictResolver::PATCHED_CODE_END
        );

        let json: Value = serde_json::from_str(json_str).unwrap();
        let mut perplexity = Vec::<String>::new();
        let prob = logprob(&json, &mut perplexity);
        assert!(prob.is_some());
        assert!(
            prob.unwrap() == -2.0,
            "wrong prob: {} expected -2.0",
            prob.unwrap()
        );
    }

    #[test]
    fn test_logprob_no_logprobs() {
        let json_str = r#"{
            "choices": [
                {
                    "message": {
                        "content": "test"
                    }
                }
            ]
        }"#;

        let json: Value = serde_json::from_str(json_str).unwrap();
        let mut perplexity = Vec::<String>::new();
        let prob = logprob(&json, &mut perplexity);
        assert!(prob.is_none());
    }

    #[test]
    fn test_logprob_empty_logprobs() {
        let json_str = r#"{
            "choices": [
                {
                    "logprobs": {
                        "content": []
                    }
                }
            ]
        }"#;

        let json: Value = serde_json::from_str(json_str).unwrap();
        let mut perplexity = Vec::<String>::new();
        let prob = logprob(&json, &mut perplexity);
        assert!(prob.is_none());
    }
}

// Local Variables:
// rust-format-on-save: t
// End:
