use camden_core::{ResolutionTier, ScanConfig, ScanSummary, ThreadingMode, move_paths, scan, write_classification_report};
use indicatif::ProgressBar;
use slint::{Image, ModelRc, SharedString, VecModel};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

slint::include_modules!();

thread_local! {
    static IMAGE_CACHE: RefCell<HashMap<PathBuf, Image>> = RefCell::new(HashMap::new());
}

#[derive(Clone)]
struct InternalFile {
    path: PathBuf,
    display_name: String,
    info: String,
    size_bytes: u64,
    sort_date: Option<String>,
    selected: bool,
    thumbnail: Option<PathBuf>,
    resolution_tier: ResolutionTier,
    moderation_tier: String,
    tags: String,
}

#[derive(Clone)]
struct InternalGroup {
    fingerprint: String,
    files: Vec<InternalFile>,
}

use std::time::Instant;

#[derive(Default, Clone)]
struct AppState {
    groups: Vec<InternalGroup>,
    scanning: bool,
    last_scan_duration: Option<std::time::Duration>,
}

fn main() -> Result<(), slint::PlatformError> {
    let ui = MainWindow::new()?;
    ui.set_root_path(default_initial_root().into());
    ui.set_target_path(default_target_path().into());
    ui.set_status_text("Select a root directory and press Scan.".into());

    let state = Arc::new(Mutex::new(AppState::default()));
    let ui_weak = ui.as_weak();

    {
        let state = Arc::clone(&state);
        let ui_weak = ui_weak.clone();
        ui.on_scan_requested(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let root_text = ui.get_root_path().trim().to_string();
                if root_text.is_empty() {
                    ui.set_status_text("Please enter a root directory before scanning.".into());
                    return;
                }

                let root_path = PathBuf::from(root_text);
                if !root_path.exists() {
                    ui.set_status_text("Root directory does not exist.".into());
                    return;
                }

                if let Ok(mut state_mut) = state.lock() {
                    state_mut.scanning = true;
                    state_mut.last_scan_duration = None;
                }
                ui.set_scanning(true);
                ui.set_progress_phase(0.0);
                ui.set_status_text("Scanning…".into());

                let rename_to_guid = ui.get_rename_to_guid();
                let detect_low_resolution = ui.get_detect_low_resolution();
                let enable_classification = ui.get_enable_classification();
                let config = build_scan_config(rename_to_guid, detect_low_resolution, enable_classification);

                let ui_weak = ui_weak.clone();
                let state = Arc::clone(&state);

                std::thread::spawn(move || perform_scan(root_path, config, state, ui_weak));
            }
        });
    }

    {
        let ui_weak = ui_weak.clone();
        ui.on_browse_root(move || {
            if let Some(folder) = rfd::FileDialog::new().pick_folder() {
                if let Some(ui) = ui_weak.upgrade() {
                    let root = folder.to_string_lossy().to_string();
                    ui.set_root_path(root.into());
                    if ui.get_target_path().is_empty() {
                        let target = folder.join("duplicates");
                        ui.set_target_path(target.to_string_lossy().to_string().into());
                    }
                }
            }
        });
    }

    {
        let ui_weak = ui_weak.clone();
        ui.on_browse_target(move || {
            if let Some(folder) = rfd::FileDialog::new().pick_folder() {
                if let Some(ui) = ui_weak.upgrade() {
                    let target = folder.to_string_lossy().to_string();
                    ui.set_target_path(target.into());
                }
            }
        });
    }

    {
        let state = Arc::clone(&state);
        let ui_weak = ui_weak.clone();
        ui.on_file_toggled(move |group_index, file_index| {
            let group_index = group_index as usize;
            let file_index = file_index as usize;

            if let Some(ui) = ui_weak.upgrade() {
                if let Ok(mut state_mut) = state.lock() {
                    if let Some(group) = state_mut.groups.get_mut(group_index) {
                        if let Some(file) = group.files.get_mut(file_index) {
                            file.selected = !file.selected;
                        }
                    }
                    let snapshot = state_mut.clone();
                    drop(state_mut);
                    refresh_ui(&ui, &snapshot, None);
                }
            }
        });
    }

    {
        let state = Arc::clone(&state);
        let ui_weak = ui_weak.clone();
        ui.on_move_requested(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let target_text = ui.get_target_path().trim().to_string();
                if target_text.is_empty() {
                    ui.set_status_text("Please choose a target directory.".into());
                    return;
                }

                let target_path = PathBuf::from(&target_text);
                let selected: Vec<PathBuf> = state
                    .lock()
                    .map(|state| {
                        state
                            .groups
                            .iter()
                            .flat_map(|group| {
                                group
                                    .files
                                    .iter()
                                    .filter(|file| file.selected)
                                    .map(|file| file.path.clone())
                            })
                            .collect()
                    })
                    .unwrap_or_default();

                if selected.is_empty() {
                    ui.set_status_text("No files selected to move.".into());
                    return;
                }

                ui.set_status_text(format!("Moving {} files…", selected.len()).into());
                let ui_weak = ui_weak.clone();
                let state = Arc::clone(&state);

                std::thread::spawn(move || perform_move(selected, target_path, state, ui_weak));
            }
        });
    }

    ui.run()
}

fn perform_scan(
    root: PathBuf,
    config: ScanConfig,
    state: Arc<Mutex<AppState>>,
    ui_weak: slint::Weak<MainWindow>,
) {
    let start_time = Instant::now();
    let classification_enabled = config.enable_classification;
    let progress_bar = Arc::new(ProgressBar::hidden());
    let summary = scan(&root, &config, &progress_bar);
    let groups = map_summary(&summary);
    let duration = start_time.elapsed();

    // Write classification report if classification was enabled
    if classification_enabled {
        if let Err(e) = write_classification_report(&summary, &root) {
            eprintln!("Failed to write classification report: {}", e);
        }
    }

    if let Ok(mut state_mut) = state.lock() {
        state_mut.groups = groups;
        state_mut.scanning = false;
        state_mut.last_scan_duration = Some(duration);
    }

    let duplicate_count = summary.duplicate_groups().count();
    let mobile_count = summary
        .groups
        .iter()
        .filter(|g| g.files.len() == 1 && g.files[0].resolution_tier == ResolutionTier::Mobile)
        .count();
    let low_res_count = summary
        .groups
        .iter()
        .filter(|g| g.files.len() == 1 && g.files[0].resolution_tier == ResolutionTier::Low)
        .count();

    // Count classified files
    let classified_count = if classification_enabled {
        summary
            .groups
            .iter()
            .flat_map(|g| &g.files)
            .filter(|f| f.moderation_tier.is_some())
            .count()
    } else {
        0
    };

    let status = if classification_enabled && classified_count > 0 {
        format!(
            "Scan complete in {:.2}s: {} duplicates, {} mobile-only, {} low-res, {} classified.",
            duration.as_secs_f64(),
            duplicate_count,
            mobile_count,
            low_res_count,
            classified_count
        )
    } else {
        format!(
            "Scan complete in {:.2}s: {} duplicates, {} mobile-only, {} low-res.",
            duration.as_secs_f64(),
            duplicate_count,
            mobile_count,
            low_res_count
        )
    };

    let state_clone = Arc::clone(&state);
    slint::invoke_from_event_loop(move || {
        if let Some(ui) = ui_weak.upgrade() {
            if let Ok(state_ref) = state_clone.lock() {
                ui.set_scanning(state_ref.scanning);
                refresh_ui(&ui, &state_ref, Some(status.clone()));
            }
        }
    })
    .ok();
}

fn perform_move(
    paths: Vec<PathBuf>,
    target: PathBuf,
    state: Arc<Mutex<AppState>>,
    ui_weak: slint::Weak<MainWindow>,
) {
    let create_target = fs::create_dir_all(&target);
    if let Err(err) = create_target {
        slint::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                ui.set_status_text(format!("Failed to create target directory: {}", err).into());
            }
        })
        .ok();
        return;
    }

    let move_result = move_paths(&paths, &target);

    let status_text = match move_result {
        Ok(stats) => {
            let moved_set: HashSet<PathBuf> = paths.into_iter().collect();
            if let Ok(mut state_mut) = state.lock() {
                for group in state_mut.groups.iter_mut() {
                    group.files.retain(|file| !moved_set.contains(&file.path));
                }
                state_mut.groups.retain(|group| !group.files.is_empty());
                ensure_largest_selected(&mut state_mut.groups);
            }
            format!("Moved {} files to {}", stats.moved, target.display())
        }
        Err(err) => format!("Failed to move files: {}", err),
    };

    let state_clone = Arc::clone(&state);
    slint::invoke_from_event_loop(move || {
        if let Some(ui) = ui_weak.upgrade() {
            if let Ok(state_ref) = state_clone.lock() {
                ui.set_scanning(state_ref.scanning);
                refresh_ui(&ui, &state_ref, Some(status_text.clone()));
            }
        }
    })
    .ok();
}

fn refresh_ui(ui: &MainWindow, state: &AppState, status_override: Option<String>) {
    let model = build_group_model(&state.groups);
    ui.set_groups(model);
    ui.set_scanning(state.scanning);
    if !state.scanning {
        ui.set_progress_phase(0.0);
    }

    let status = status_override.unwrap_or_else(|| format_status(state));
    ui.set_status_text(status.into());
}

fn build_group_model(groups: &[InternalGroup]) -> ModelRc<GroupData> {
    let group_data: Vec<GroupData> = groups
        .iter()
        .map(|group| {
            let files: Vec<FileData> = group
                .files
                .iter()
                .map(|file| FileData {
                    display_name: SharedString::from(file.display_name.clone()),
                    info: SharedString::from(file.info.clone()),
                    selected: file.selected,
                    thumbnail: file
                        .thumbnail
                        .as_ref()
                        .and_then(|path| load_thumbnail(path))
                        .unwrap_or_default(),
                    resolution_tier: match file.resolution_tier {
                        ResolutionTier::High => 0,
                        ResolutionTier::Mobile => 1,
                        ResolutionTier::Low => 2,
                    },
                    moderation_tier: SharedString::from(file.moderation_tier.clone()),
                    tags: SharedString::from(file.tags.clone()),
                })
                .collect();
            GroupData {
                fingerprint: SharedString::from(group.fingerprint.clone()),
                file_count: group.files.len() as i32,
                files: ModelRc::from(Rc::new(VecModel::from(files))),
            }
        })
        .collect();

    ModelRc::from(Rc::new(VecModel::from(group_data)))
}

fn map_summary(summary: &ScanSummary) -> Vec<InternalGroup> {
    summary
        .actionable_groups()
        .map(|group| {
            let fingerprint = format!("{:016x}", group.fingerprint);
            let mut files: Vec<InternalFile> = group
                .files
                .iter()
                .map(|file| {
                    let display_name = file
                        .path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or("unknown")
                        .to_string();
                    let info = format_file_info(file);
                    let sort_date = file.captured_at.clone().or_else(|| file.modified.clone());

                    InternalFile {
                        path: file.path.clone(),
                        display_name,
                        info,
                        size_bytes: file.size_bytes,
                        sort_date,
                        selected: false,
                        thumbnail: file.thumbnail.clone(),
                        resolution_tier: file.resolution_tier,
                        moderation_tier: file.moderation_tier.clone().unwrap_or_default(),
                        tags: file.tags.join(", "),
                    }
                })
                .collect();

            // For duplicate groups: select all except the best one to keep
            // For resolution singletons: pre-select only if Low tier
            if files.len() > 1 {
                let keep_index = find_keep_index(&files);
                for (index, file) in files.iter_mut().enumerate() {
                    file.selected = index != keep_index;
                }
            } else if files.len() == 1 && files[0].resolution_tier.should_preselect() {
                files[0].selected = true;
            }

            InternalGroup { fingerprint, files }
        })
        .collect()
}

fn find_keep_index(files: &[InternalFile]) -> usize {
    if files.is_empty() {
        return 0;
    }

    let max_size = files.iter().map(|f| f.size_bytes).max().unwrap_or(0);

    files
        .iter()
        .enumerate()
        .filter(|(_, f)| f.size_bytes == max_size)
        .max_by(|(_, a), (_, b)| {
            let date_cmp = a.sort_date.cmp(&b.sort_date);
            if date_cmp != std::cmp::Ordering::Equal {
                return date_cmp;
            }
            b.path.cmp(&a.path)
        })
        .map(|(index, _)| index)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn make_file(name: &str, size: u64, date: Option<&str>) -> InternalFile {
        InternalFile {
            path: PathBuf::from(name),
            display_name: name.to_string(),
            info: String::new(),
            size_bytes: size,
            sort_date: date.map(|s| s.to_string()),
            selected: false,
            thumbnail: None,
            resolution_tier: ResolutionTier::High,
            moderation_tier: String::new(),
            tags: String::new(),
        }
    }

    #[test]
    fn keep_largest() {
        let files = vec![
            make_file("small.jpg", 100, Some("2023-01-01")),
            make_file("large.jpg", 200, Some("2023-01-01")),
        ];
        assert_eq!(find_keep_index(&files), 1);
    }

    #[test]
    fn keep_newest_when_size_same() {
        let files = vec![
            make_file("old.jpg", 100, Some("2023-01-01")),
            make_file("new.jpg", 100, Some("2023-01-02")),
        ];
        assert_eq!(find_keep_index(&files), 1);
    }

    #[test]
    fn keep_alphabetical_when_size_and_date_same() {
        // "b.jpg" should be kept because we use b.path.cmp(&a.path) which is reverse lexical sort?
        // Wait, the logic is `b.path.cmp(&a.path)`.
        // If a="a.jpg", b="b.jpg".
        // a.cmp(b) is Less. b.cmp(a) is Greater.
        // max_by will pick the one that is Greater.
        // So "b.jpg" > "a.jpg" in string comparison? Yes.
        // So b.path.cmp(&a.path):
        // compare(current_max, candidate).
        // if candidate > current_max, candidate becomes max.
        // Let's trace.
        // Logic: .max_by(|(_, a), (_, b)| ... b.path.cmp(&a.path))
        // This is tricky. max_by returns the element that yields Ordering::Greater when compared to others.
        // The closure compares `a` (left) and `b` (right).
        // If we want "a" to be "greater" (selected) than "b", we should return Greater.
        // We want stable sort order? Usually "path" tiebreaker is purely deterministic.
        // Let's assume we want "first" file alphabetically if all else equal?
        // If we want "a.jpg" to be kept over "b.jpg", "a" should be "better".
        // If b.path.cmp(&a.path) is used:
        // "a.jpg" vs "b.jpg" -> "b".cmp("a") -> Greater. So "a" is Greater than "b"? No.
        // max_by(cmp): if cmp(a, b) == Greater, a is max.
        // cmp("a", "b") -> "b".cmp("a") -> Greater.
        // So "a" is considered "greater" (better) than "b".
        // So "a.jpg" should be kept.
        // Let's test this assumption.

        let files = vec![
            make_file("a.jpg", 100, Some("2023-01-01")),
            make_file("b.jpg", 100, Some("2023-01-01")),
        ];
        assert_eq!(find_keep_index(&files), 0); // a.jpg
    }

    #[test]
    fn keep_largest_ignores_date() {
        let files = vec![
            make_file("small_new.jpg", 100, Some("2023-01-02")),
            make_file("large_old.jpg", 200, Some("2023-01-01")),
        ];
        assert_eq!(find_keep_index(&files), 1);
    }

    #[test]
    fn handle_missing_dates() {
        let files = vec![
            make_file("no_date.jpg", 100, None),
            make_file("with_date.jpg", 100, Some("2023-01-01")),
        ];
        // Some("...") > None is true for Option.
        // So with_date should be kept.
        assert_eq!(find_keep_index(&files), 1);
    }
}


fn ensure_largest_selected(groups: &mut [InternalGroup]) {
    for group in groups.iter_mut() {
        if group.files.is_empty() {
            continue;
        }
        let keep_index = find_keep_index(&group.files);
        for (index, file) in group.files.iter_mut().enumerate() {
            file.selected = index != keep_index;
        }
    }
}

fn format_file_info(file: &camden_core::DuplicateEntry) -> String {
    let size = if file.size_bytes == 0 {
        "0 B".to_string()
    } else {
        format_size(file.size_bytes)
    };
    let dimensions = format!("{}×{}", file.dimensions.0, file.dimensions.1);
    let timestamp = file
        .captured_at
        .as_deref()
        .or_else(|| file.modified.as_deref())
        .unwrap_or("unknown");
    format!("{} • {} • {}", size, dimensions, timestamp)
}

fn format_size(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KB", "MB", "GB"];
    let mut size = bytes as f64;
    let mut unit = 0usize;
    while size >= 1024.0 && unit < UNITS.len() - 1 {
        size /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{} {}", bytes, UNITS[unit])
    } else {
        format!("{:.1} {}", size, UNITS[unit])
    }
}

fn format_status(state: &AppState) -> String {
    if state.scanning {
        return "Scanning…".to_string();
    }
    let group_count = state.groups.len();
    let mut file_count = 0usize;
    let mut selected_count = 0usize;
    for group in &state.groups {
        file_count += group.files.len();
        selected_count += group.files.iter().filter(|file| file.selected).count();
    }
    
    let time_info = if let Some(d) = state.last_scan_duration {
        format!(" • Time: {:.2}s", d.as_secs_f64())
    } else {
        String::new()
    };

    format!(
        "Groups: {} • Files: {} • Selected: {}{}",
        group_count, file_count, selected_count, time_info
    )
}

fn build_scan_config(rename_to_guid: bool, detect_low_resolution: bool, enable_classification: bool) -> ScanConfig {
    let mut config = ScanConfig::new(default_extensions(), ThreadingMode::Parallel);
    if let Some(mut dir) = dirs::data_local_dir() {
        dir.push("Camden");
        dir.push("thumbnails");
        config = config.with_thumbnail_root(dir);
    }
    config
        .with_guid_rename(rename_to_guid)
        .with_low_resolution_detection(detect_low_resolution)
        .with_classification(enable_classification)
}

fn default_extensions() -> Vec<String> {
    ["jpg", "jpeg", "png", "gif", "bmp", "webp"]
        .iter()
        .map(|ext| ext.to_string())
        .collect()
}

fn default_initial_root() -> String {
    dirs::picture_dir()
        .or_else(|| dirs::home_dir())
        .unwrap_or_else(|| PathBuf::from("C:\\"))
        .to_string_lossy()
        .to_string()
}

fn default_target_path() -> String {
    dirs::data_local_dir()
        .map(|mut dir| {
            dir.push("Camden");
            dir.push("duplicates");
            dir.to_string_lossy().to_string()
        })
        .unwrap_or_else(|| String::from("C:\\Camden\\duplicates"))
}

fn load_thumbnail(path: &Path) -> Option<Image> {
    IMAGE_CACHE.with(|cache_cell| {
        {
            let cache = cache_cell.borrow();
            if let Some(image) = cache.get(path) {
                return Some(image.clone());
            }
        }

        let mut cache = cache_cell.borrow_mut();
        if let Ok(image) = Image::load_from_path(path) {
            cache.insert(path.to_path_buf(), image.clone());
            return Some(image);
        }

        if path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("webp"))
            .unwrap_or(false)
        {
            let png_path = path.with_extension("png");
            if let Some(image) = cache.get(&png_path) {
                let image = image.clone();
                cache.insert(path.to_path_buf(), image.clone());
                return Some(image);
            }
            if let Ok(image) = Image::load_from_path(&png_path) {
                cache.insert(png_path.clone(), image.clone());
                cache.insert(path.to_path_buf(), image.clone());
                return Some(image);
            }
        }

        None
    })
}

