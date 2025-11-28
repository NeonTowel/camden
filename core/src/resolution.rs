//! Low-resolution image detection utilities.
//!
//! Resolution is determined by a single dimension only:
//! - Landscape/square: only **width** matters (height is irrelevant)
//! - Portrait: only **height** matters (width is irrelevant)

use serde::{Deserialize, Serialize};

/// Minimum width for landscape images to be considered "high resolution".
pub const MIN_LANDSCAPE_WIDTH: i32 = 1200;

/// Minimum height for portrait images to be considered "high resolution" (desktop).
pub const MIN_PORTRAIT_HEIGHT_DESKTOP: i32 = 1200;

/// Minimum height for portrait images to be usable on mobile screens.
pub const MIN_PORTRAIT_HEIGHT_MOBILE: i32 = 850;

/// Resolution classification for an image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ResolutionTier {
    /// Image meets desktop resolution requirements.
    #[default]
    High,
    /// Portrait image suitable for mobile but not desktop (850-1199px height).
    Mobile,
    /// Image is too low resolution even for mobile use.
    Low,
}

impl ResolutionTier {
    /// Returns true if the image should be flagged in the UI.
    pub fn is_actionable(self) -> bool {
        matches!(self, ResolutionTier::Mobile | ResolutionTier::Low)
    }

    /// Returns true if the image should be pre-selected for moving.
    pub fn should_preselect(self) -> bool {
        matches!(self, ResolutionTier::Low)
    }
}

/// Determines the resolution tier for an image based on its dimensions.
///
/// - **Landscape/square** (width >= height): 
///   - High if width >= 1200, else Low
/// - **Portrait** (height > width):
///   - High if height >= 1200
///   - Mobile if height >= 850 (usable on mobile screens)
///   - Low if height < 850
pub fn resolution_tier(width: i32, height: i32) -> ResolutionTier {
    if width >= height {
        // Landscape or square: single tier based on width
        if width >= MIN_LANDSCAPE_WIDTH {
            ResolutionTier::High
        } else {
            ResolutionTier::Low
        }
    } else {
        // Portrait: two-tier based on height
        if height >= MIN_PORTRAIT_HEIGHT_DESKTOP {
            ResolutionTier::High
        } else if height >= MIN_PORTRAIT_HEIGHT_MOBILE {
            ResolutionTier::Mobile
        } else {
            ResolutionTier::Low
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn landscape_high_resolution() {
        assert_eq!(resolution_tier(1200, 800), ResolutionTier::High);
        assert_eq!(resolution_tier(1920, 1080), ResolutionTier::High);
        assert_eq!(resolution_tier(3840, 2160), ResolutionTier::High);
    }

    #[test]
    fn landscape_low_resolution() {
        assert_eq!(resolution_tier(1199, 800), ResolutionTier::Low);
        assert_eq!(resolution_tier(1080, 720), ResolutionTier::Low);
        assert_eq!(resolution_tier(800, 600), ResolutionTier::Low);
    }

    #[test]
    fn landscape_ignores_height() {
        // Width is 1200 (OK), height varies - all should be high
        assert_eq!(resolution_tier(1200, 100), ResolutionTier::High);
        assert_eq!(resolution_tier(1200, 500), ResolutionTier::High);
        assert_eq!(resolution_tier(1200, 800), ResolutionTier::High);
        // Width is 1100 (low), height varies - all should be low
        assert_eq!(resolution_tier(1100, 100), ResolutionTier::Low);
        assert_eq!(resolution_tier(1100, 720), ResolutionTier::Low);
    }

    #[test]
    fn portrait_high_resolution() {
        assert_eq!(resolution_tier(800, 1200), ResolutionTier::High);
        assert_eq!(resolution_tier(1080, 1920), ResolutionTier::High);
    }

    #[test]
    fn portrait_mobile_resolution() {
        assert_eq!(resolution_tier(600, 850), ResolutionTier::Mobile);
        assert_eq!(resolution_tier(720, 1000), ResolutionTier::Mobile);
        assert_eq!(resolution_tier(800, 1199), ResolutionTier::Mobile);
    }

    #[test]
    fn portrait_low_resolution() {
        assert_eq!(resolution_tier(400, 849), ResolutionTier::Low);
        assert_eq!(resolution_tier(480, 640), ResolutionTier::Low);
    }

    #[test]
    fn portrait_ignores_width() {
        // Height is 1200 (high), width varies
        assert_eq!(resolution_tier(100, 1200), ResolutionTier::High);
        assert_eq!(resolution_tier(800, 1200), ResolutionTier::High);
        // Height is 1000 (mobile), width varies
        assert_eq!(resolution_tier(100, 1000), ResolutionTier::Mobile);
        assert_eq!(resolution_tier(600, 1000), ResolutionTier::Mobile);
        // Height is 800 (low), width varies
        assert_eq!(resolution_tier(100, 800), ResolutionTier::Low);
        assert_eq!(resolution_tier(500, 800), ResolutionTier::Low);
    }

    #[test]
    fn square_uses_landscape_rule() {
        assert_eq!(resolution_tier(1000, 1000), ResolutionTier::Low);
        assert_eq!(resolution_tier(1200, 1200), ResolutionTier::High);
    }

    #[test]
    fn tier_actionable() {
        assert!(!ResolutionTier::High.is_actionable());
        assert!(ResolutionTier::Mobile.is_actionable());
        assert!(ResolutionTier::Low.is_actionable());
    }

    #[test]
    fn tier_preselect() {
        assert!(!ResolutionTier::High.should_preselect());
        assert!(!ResolutionTier::Mobile.should_preselect());
        assert!(ResolutionTier::Low.should_preselect());
    }
}
