//! Utilities for interpreting model outputs into moderation results.
//!
//! This module provides common functions for validating probabilities,
//! mapping scores to tiers, and creating synthetic category scores.

use super::{ModerationCategories, ModerationTier};
use crate::classifier::ClassifierError;

/// Validate that a probability vector has the expected length.
///
/// # Arguments
///
/// * `probs` - The probability vector to validate
/// * `expected` - The expected number of classes
/// * `format` - Format name for error messages
///
/// # Returns
///
/// * `Ok(())` if validation passes
/// * `Err(ClassifierError)` with descriptive message if validation fails
///
/// # Examples
///
/// ```ignore
/// validate_probabilities(&[0.1, 0.9], 2, "Binary")?;
/// ```
pub fn validate_probabilities(
    probs: &[f32],
    expected: usize,
    format: &str,
) -> Result<(), ClassifierError> {
    if probs.len() < expected {
        return Err(ClassifierError::Processing(format!(
            "expected {} classes for {}, got {}",
            expected,
            format,
            probs.len()
        )));
    }
    Ok(())
}

/// Map a single confidence score to a moderation tier using standard thresholds.
///
/// The standard thresholds are:
/// - `> 0.8`: Restricted (high risk)
/// - `> 0.5`: Mature (medium risk)
/// - `> 0.3`: Sensitive (low risk)
/// - `<= 0.3`: Safe
///
/// # Arguments
///
/// * `score` - Confidence score (0.0-1.0)
///
/// # Returns
///
/// The appropriate `ModerationTier` for the score
///
/// # Examples
///
/// ```ignore
/// let tier = score_to_tier(0.85); // Returns ModerationTier::Restricted
/// let tier = score_to_tier(0.2);  // Returns ModerationTier::Safe
/// ```
pub fn score_to_tier(score: f32) -> ModerationTier {
    if score > 0.8 {
        ModerationTier::Restricted
    } else if score > 0.5 {
        ModerationTier::Mature
    } else if score > 0.3 {
        ModerationTier::Sensitive
    } else {
        ModerationTier::Safe
    }
}

/// Create synthetic category scores based on a tier and available scores.
///
/// This is used when a model outputs tier-based classifications (Safe, Mild, Explicit)
/// but doesn't provide per-category breakdowns. We create synthetic categories that
/// reflect the overall tier assessment.
///
/// # Arguments
///
/// * `tier` - The determined moderation tier
/// * `explicit_score` - Score for explicit/restricted content (if available)
/// * `mild_score` - Score for mild/sensitive content (if available)
/// * `safe_score` - Score for safe content (if available)
///
/// # Returns
///
/// A `ModerationCategories` struct with synthetic scores
///
/// # Examples
///
/// ```ignore
/// let categories = synthetic_categories(
///     ModerationTier::Restricted,
///     0.9,   // explicit_score
///     0.05,  // mild_score
///     0.05,  // safe_score
/// );
/// ```
#[allow(dead_code)]
pub fn synthetic_categories(
    tier: ModerationTier,
    explicit_score: f32,
    mild_score: f32,
    safe_score: f32,
) -> ModerationCategories {
    match tier {
        ModerationTier::Restricted => {
            // High explicit content - likely porn or highly suggestive
            ModerationCategories {
                porn: explicit_score.max(0.7),
                sexy: mild_score.max(0.2),
                hentai: 0.0,
                drawings: 0.0,
                neutral: safe_score,
            }
        }
        ModerationTier::Mature => {
            // Moderate risk - likely sexy/suggestive content
            ModerationCategories {
                porn: explicit_score.min(0.5),
                sexy: mild_score.max(0.6),
                hentai: 0.0,
                drawings: 0.0,
                neutral: safe_score,
            }
        }
        ModerationTier::Sensitive => {
            // Low risk - mildly suggestive or artistic nudity
            ModerationCategories {
                porn: 0.0,
                sexy: mild_score.max(0.4),
                hentai: 0.0,
                drawings: 0.0,
                neutral: safe_score.max(0.5),
            }
        }
        ModerationTier::Safe => {
            // Safe content
            ModerationCategories {
                porn: 0.0,
                sexy: 0.0,
                hentai: 0.0,
                drawings: 0.0,
                neutral: safe_score.max(0.9),
            }
        }
    }
}

/// Create fully synthetic categories when only a tier is available.
///
/// This is a simplified version of `synthetic_categories` that generates
/// reasonable category scores based solely on the tier, without requiring
/// the original model probabilities.
///
/// # Arguments
///
/// * `tier` - The determined moderation tier
///
/// # Returns
///
/// A `ModerationCategories` struct with synthetic scores
#[allow(dead_code)]
pub fn synthetic_categories_from_tier(tier: ModerationTier) -> ModerationCategories {
    match tier {
        ModerationTier::Restricted => ModerationCategories {
            porn: 0.85,
            sexy: 0.1,
            hentai: 0.0,
            drawings: 0.0,
            neutral: 0.05,
        },
        ModerationTier::Mature => ModerationCategories {
            porn: 0.3,
            sexy: 0.65,
            hentai: 0.0,
            drawings: 0.0,
            neutral: 0.05,
        },
        ModerationTier::Sensitive => ModerationCategories {
            porn: 0.0,
            sexy: 0.4,
            hentai: 0.0,
            drawings: 0.0,
            neutral: 0.6,
        },
        ModerationTier::Safe => ModerationCategories {
            porn: 0.0,
            sexy: 0.0,
            hentai: 0.0,
            drawings: 0.0,
            neutral: 0.95,
        },
    }
}
