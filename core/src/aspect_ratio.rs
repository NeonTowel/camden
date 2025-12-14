//! Utilities for calculating and evaluating image aspect ratios.

/// A list of standard display and photographic aspect ratios.
/// Ratios are represented as `(width, height)`.
const STANDARD_ASPECT_RATIOS: &[(f32, f32)] = &[
    (16.0, 9.0),   // Widescreen video
    (4.0, 3.0),    // Standard TV / Monitor
    (3.0, 2.0),    // 35mm film
    (1.0, 1.0),    // Square
    (16.0, 10.0),  // Widescreen computer displays
    (5.0, 4.0),    // Common for larger format photography
    (21.0, 9.0),   // Ultrawide cinema
];

/// The tolerance to use when comparing aspect ratios.
/// An aspect ratio is considered "standard" if it is within this tolerance
/// of a known standard ratio.
const ASPECT_RATIO_TOLERANCE: f32 = 0.05;

/// Checks if a given aspect ratio is close to one of the standard ratios.
///
/// # Arguments
///
/// * `width` - The width of the image.
/// * `height` - The height of the image.
///
/// # Returns
///
/// `true` if the aspect ratio is considered standard, `false` otherwise.
pub fn is_standard_aspect_ratio(width: i32, height: i32) -> bool {
    if width == 0 || height == 0 {
        return false;
    }

    let aspect_ratio = width as f32 / height as f32;

    for &(standard_w, standard_h) in STANDARD_ASPECT_RATIOS {
        let standard_ratio = standard_w / standard_h;
        if (aspect_ratio - standard_ratio).abs() < ASPECT_RATIO_TOLERANCE {
            return true;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_standard_aspect_ratio() {
        // Exact 16:9
        assert!(is_standard_aspect_ratio(1920, 1080));
        // Close to 16:9
        assert!(is_standard_aspect_ratio(1921, 1080));
        // Not standard
        assert!(!is_standard_aspect_ratio(1000, 999));
        // Exact 4:3
        assert!(is_standard_aspect_ratio(1024, 768));
        // Exact 3:2
        assert!(is_standard_aspect_ratio(1080, 720));
        // Exact 1:1
        assert!(is_standard_aspect_ratio(1000, 1000));
        // Portrait orientation (9:16) should also be standard
        assert!(is_standard_aspect_ratio(1080, 1920));
        // Zero width or height
        assert!(!is_standard_aspect_ratio(0, 1080));
        assert!(!is_standard_aspect_ratio(1920, 0));
    }
}