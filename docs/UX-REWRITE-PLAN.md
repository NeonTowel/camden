# Camden UX Rewrite Plan

## Overview

This document tracks the redesign of Camden's Slint frontend from a single-screen layout to a multi-tab, filter-driven photo management application.

## Design Decisions

| Aspect | Decision |
|--------|----------|
| Navigation | Top tabs (Scan, Duplicates, Organize, Settings) |
| Duplicates | Group grid with "Keep" indicators |
| Filtering | Sidebar filters (always visible checkboxes) |
| Sensitive content | Unobtrusive badges (no blur) |
| Archive behavior | Move to archive folder (recoverable) |
| Theme | Light/Dark mode support |

---

## Phase 1: Component Architecture (COMPLETED)

### New File Structure
```
camden-frontend/ui/
├── main.slint                    # Main window with tab navigation
├── components/
│   ├── theme.slint               # Theme global with dark/light color palettes
│   ├── badge.slint               # ResolutionBadge, ModerationBadge, IconBadge
│   ├── photo-card.slint          # PhotoCard component with PhotoData struct
│   ├── filter-sidebar.slint      # Checkbox filters for gallery
│   ├── tab-bar.slint             # Horizontal tab navigation
│   └── progress-bar.slint        # Scan progress indicator
└── views/
    ├── scan-view.slint           # Folder selection, options, scan progress
    ├── duplicates-view.slint     # Duplicate group grid with stats
    ├── gallery-view.slint        # Filtered photo gallery
    └── settings-view.slint       # App configuration
```

### Key Data Structures (Slint)
```slint
// Photo card data
struct PhotoData {
    id: int,
    display_name: string,
    info: string,
    thumbnail: image,
    selected: bool,
    resolution_tier: int,      // 0=High, 1=Mobile, 2=Low
    orientation: int,          // 0=Landscape, 1=Portrait, 2=Square
    moderation_tier: string,   // "", "Safe", "Sensitive", "Mature", "Restricted"
    tags: string,
    is_keep_candidate: bool,
    group_id: int,
}

// Duplicate group
struct DuplicateGroup {
    fingerprint: string,
    files: [PhotoData],
    total_size_bytes: int,
    reclaimable_bytes: int,
}

// Stats structures
struct DuplicateStats { total_groups, total_files, total_duplicates, reclaimable_mb, selected_count }
struct GalleryStats { total_photos, filtered_photos, selected_count, landscape_count, portrait_count, square_count }
```

### Callbacks Defined
```
// Scan
browse_root(), browse_target(), scan_requested()

// Duplicates
file_toggled(group_idx, file_idx), select_all_best(), archive_selected()
select_best_in_group(idx), archive_others_in_group(idx)

// Gallery
gallery_photo_clicked(idx), gallery_photo_toggle_selected(idx)
gallery_filter_changed(), gallery_export_selected(), gallery_archive_selected()
gallery_select_all(), gallery_deselect_all()

// Settings
browse_archive_path(), browse_cache_path(), clear_cache()
save_settings(), reset_defaults()
```

---

## Phase 2: Theme Support (COMPLETED)

### Implementation
- Created `theme.slint` with centralized `ThemeColors` struct
- `Theme` global with reactive `dark_mode` property
- All components import and use `Theme.colors.*`
- Dark mode toggle in Settings > Display Preferences

### Theme Architecture
```slint
// theme.slint
export struct ThemeColors {
    bg_primary, bg_secondary, bg_card, bg_hover, bg_input: color,
    text_primary, text_secondary, text_muted: color,
    border_default, border_focus: color,
    accent_primary, accent_success, accent_warning, accent_danger: color,
    statusbar_bg, statusbar_text: color,
    tab_bg, tab_active_bg, tab_indicator: color,
}

export global Theme {
    in-out property <bool> dark_mode: true;
    out property <ThemeColors> colors: dark_mode ? DarkTheme.colors : LightTheme.colors;
    // Badge colors (consistent across themes)
    out property <color> badge_high_res: #10b981;
    out property <color> badge_mobile_res: #f59e0b;
    out property <color> badge_low_res: #ef4444;
    // ... etc
}
```

### Color Palette (Actual)
```
Dark Theme:
- bg_primary: #1a1a2e (deep navy)
- bg_secondary: #16213e (dark blue)
- bg_card: #1f2937 (slate gray)
- bg_hover: #374151
- text_primary: #f3f4f6
- text_secondary: #9ca3af
- text_muted: #6b7280
- border_default: #374151
- accent_primary: #60a5fa (cyan blue)
- accent_success: #10b981 (emerald)
- accent_warning: #f59e0b (amber)
- accent_danger: #ef4444 (red)

Light Theme:
- bg_primary: #f8f9fa
- bg_secondary: #e9ecef
- bg_card: #ffffff
- text_primary: #1f2937
- text_secondary: #4b5563
- accent_primary: #3b82f6 (blue)
```

### Files Updated for Theme
- [x] main.slint
- [x] tab-bar.slint
- [x] badge.slint
- [x] photo-card.slint
- [x] progress-bar.slint
- [x] filter-sidebar.slint
- [x] scan-view.slint
- [x] duplicates-view.slint
- [x] gallery-view.slint
- [x] settings-view.slint

---

## Phase 3: Backend Wiring (TODO)

### Overview
Connect the new Slint UI to the existing camden-core scan/classify functionality.

### Step 1: Update main.rs Structure
```rust
// camden-frontend/src/main.rs

// Key state to maintain:
struct AppState {
    all_photos: Vec<PhotoData>,        // All scanned photos
    duplicate_groups: Vec<DuplicateGroup>,
    gallery_photos: Vec<PhotoData>,    // Filtered view
    scan_in_progress: bool,
    archive_path: PathBuf,
}
```

**Tasks:**
- [ ] Add `slint::SharedString` conversions for PhotoData
- [ ] Import new UI types from generated Slint bindings
- [ ] Set up state management (Arc<Mutex<AppState>> or similar)

### Step 2: Implement Scan Callbacks
```rust
// Connect to existing camden-core scanner
ui.on_start_scan(move || {
    let root = ui.get_root_path().to_string();
    let config = ScanConfig::new(IMAGE_EXTENSIONS.to_vec(), ThreadingMode::Parallel)
        .with_classification(ui.get_classify_images())
        .with_low_resolution_detection(ui.get_detect_low_res());

    // Run scan in background thread
    std::thread::spawn(move || {
        let scanner = Scanner::new(&root, config);
        for result in scanner.scan() {
            // Update progress UI
            ui_handle.upgrade_in_event_loop(|ui| {
                ui.set_files_found(count);
                ui.set_current_file(file.to_string().into());
            });
        }
        // Convert results to PhotoData
        // ...
    });
});
```

**Tasks:**
- [ ] Wire `browse_root()` → native file dialog
- [ ] Wire `browse_archive()` → native file dialog
- [ ] Wire `start_scan()` → spawn scanner thread
- [ ] Update progress properties during scan
- [ ] Convert `DuplicateEntry` → `PhotoData`
- [ ] Populate `duplicate_groups` property
- [ ] Calculate and set `DuplicateStats`

### Step 3: PhotoData Conversion
```rust
fn scan_result_to_photo_data(entry: &DuplicateEntry, idx: i32) -> PhotoData {
    PhotoData {
        id: idx,
        display_name: entry.path.file_name().unwrap().to_string_lossy().into(),
        info: format!("{} • {}x{}", format_size(entry.size), entry.width, entry.height).into(),
        thumbnail: load_thumbnail(&entry.path), // slint::Image
        selected: false,
        resolution_tier: classify_resolution(entry.width, entry.height),
        orientation: classify_orientation(entry.width, entry.height),
        moderation_tier: entry.moderation_tier.clone().unwrap_or_default().into(),
        tags: entry.tags.join(", ").into(),
        is_keep_candidate: false,
        group_id: -1,
    }
}

fn classify_resolution(w: u32, h: u32) -> i32 {
    let min_dim = w.min(h);
    if min_dim >= 1200 { 0 }      // High
    else if min_dim >= 850 { 1 }  // Mobile
    else { 2 }                     // Low
}

fn classify_orientation(w: u32, h: u32) -> i32 {
    if w > h { 0 }        // Landscape
    else if h > w { 1 }   // Portrait
    else { 2 }            // Square
}
```

### Step 4: Gallery Filtering (Rust-side)
```rust
fn apply_gallery_filters(
    photos: &[PhotoData],
    show_landscape: bool,
    show_portrait: bool,
    show_square: bool,
    show_high_res: bool,
    show_mobile_res: bool,
    show_low_res: bool,
    show_safe: bool,
    show_sensitive: bool,
    show_mature: bool,
    show_restricted: bool,
    tag_search: &str,
) -> Vec<PhotoData> {
    photos.iter()
        .filter(|p| match p.orientation {
            0 => show_landscape,
            1 => show_portrait,
            2 => show_square,
            _ => true,
        })
        .filter(|p| match p.resolution_tier {
            0 => show_high_res,
            1 => show_mobile_res,
            2 => show_low_res,
            _ => true,
        })
        .filter(|p| {
            let tier = p.moderation_tier.as_str();
            match tier {
                "" | "Safe" => show_safe,
                "Sensitive" => show_sensitive,
                "Mature" => show_mature,
                "Restricted" => show_restricted,
                _ => true,
            }
        })
        .filter(|p| {
            tag_search.is_empty() ||
            p.tags.to_lowercase().contains(&tag_search.to_lowercase())
        })
        .cloned()
        .collect()
}
```

**Tasks:**
- [ ] Wire `filter_changed()` callback
- [ ] Re-apply filters when any checkbox changes
- [ ] Update `gallery_photos` property with filtered results
- [ ] Recalculate `GalleryStats`

### Step 5: Duplicate Management Actions
```rust
// Select best in group (highest resolution, largest file)
ui.on_select_best_in_group(move |group_idx| {
    // Find best candidate in group
    // Set is_keep_candidate = true for best
    // Set selected = true for others (to archive)
});

// Archive selected files
ui.on_archive_selected(move || {
    let archive_path = ui.get_settings_archive_path();
    for photo in selected_photos {
        std::fs::rename(&photo.path, archive_path.join(&photo.display_name))?;
    }
    // Remove from lists, update stats
});
```

**Tasks:**
- [ ] Wire `select_best_in_group(idx)`
- [ ] Wire `archive_others_in_group(idx)`
- [ ] Wire `select_all_best()`
- [ ] Wire `archive_selected()`
- [ ] Implement file move to archive folder
- [ ] Update UI state after archive

### Step 6: Thumbnail Loading
```rust
fn load_thumbnail(path: &Path) -> slint::Image {
    // Option 1: Use existing thumbnail cache from camden-core
    if let Some(thumb) = ThumbnailCache::get(path) {
        return slint::Image::from_rgba8(thumb);
    }

    // Option 2: Generate on-the-fly
    let img = image::open(path).ok()?;
    let thumb = img.thumbnail(150, 150);
    slint::Image::from_rgba8(slint::SharedPixelBuffer::clone_from_slice(
        thumb.to_rgba8().as_raw(),
        thumb.width(),
        thumb.height(),
    ))
}
```

**Tasks:**
- [ ] Integrate with existing `ThumbnailCache` from camden-core
- [ ] Handle missing/corrupt images gracefully
- [ ] Consider lazy loading for large galleries

### Step 7: Settings Persistence
```rust
// Load/save settings to config file
#[derive(Serialize, Deserialize)]
struct AppSettings {
    archive_path: String,
    cache_path: String,
    show_tags: bool,
    compact_cards: bool,
    dark_mode: bool,
}

fn load_settings() -> AppSettings { ... }
fn save_settings(settings: &AppSettings) { ... }
```

**Tasks:**
- [ ] Wire `save_settings()` → serialize to file
- [ ] Wire `reset_defaults()` → restore defaults
- [ ] Wire `clear_cache()` → delete thumbnail cache
- [ ] Load settings on startup
- [ ] Persist dark_mode preference

### Data Flow Diagram
```
                    ┌─────────────────┐
                    │   User Action   │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌─────────┐    ┌──────────┐   ┌──────────┐
        │  Scan   │    │  Filter  │   │  Archive │
        └────┬────┘    └────┬─────┘   └────┬─────┘
             │              │              │
             ▼              │              │
    ┌────────────────┐      │              │
    │  camden-core   │      │              │
    │    Scanner     │      │              │
    └───────┬────────┘      │              │
            │               │              │
            ▼               ▼              ▼
    ┌───────────────────────────────────────────┐
    │              AppState (Rust)              │
    │  - all_photos: Vec<PhotoData>             │
    │  - duplicate_groups: Vec<DuplicateGroup>  │
    │  - gallery_photos: Vec<PhotoData>         │
    └───────────────────┬───────────────────────┘
                        │
                        ▼
    ┌───────────────────────────────────────────┐
    │            Slint UI Properties            │
    │  - duplicate_groups                       │
    │  - duplicate_stats                        │
    │  - gallery_photos                         │
    │  - gallery_stats                          │
    └───────────────────────────────────────────┘
```

---

## Phase 4: Polish (TODO)

- [ ] Fix padding warnings (use HorizontalLayout instead of Rectangle)
- [ ] Add keyboard shortcuts (Ctrl+A select all, Delete to archive)
- [ ] Photo preview modal (click to enlarge)
- [ ] Drag selection in gallery
- [ ] Undo snackbar for archive operations
- [ ] Remember last scan folder
- [ ] Export to folder dialog

---

## Migration Notes

### Legacy Compatibility
The following legacy types are preserved for backward compatibility:
- `FileData` - Original file data struct
- `GroupData` - Original group struct
- `groups` property - Original groups array

These can be removed once backend is fully migrated to new data structures.

### Breaking Changes from Old UI
- Window size increased: 1024x760 → 1200x800
- Single screen → Multi-tab navigation
- Results view → Split into Duplicates and Organize tabs
- Move to target → Archive to configured folder
