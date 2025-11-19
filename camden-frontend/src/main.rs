use camden_core::{ScanConfig, ScanSummary, ThreadingMode, move_paths, scan};
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
    selected: bool,
    thumbnail: Option<PathBuf>,
}

#[derive(Clone)]
struct InternalGroup {
    fingerprint: String,
    files: Vec<InternalFile>,
}

#[derive(Default, Clone)]
struct AppState {
    groups: Vec<InternalGroup>,
    scanning: bool,
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
                }
                ui.set_scanning(true);
                ui.set_progress_phase(0.0);
                ui.set_status_text("Scanning…".into());
                let config = build_scan_config();
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
    let progress_bar = Arc::new(ProgressBar::hidden());
    let summary = scan(&root, &config, &progress_bar);
    let groups = map_summary(&summary);

    if let Ok(mut state_mut) = state.lock() {
        state_mut.groups = groups;
        state_mut.scanning = false;
    }

    let status = format!(
        "Scan complete: {} duplicate groups, {} files.",
        summary.duplicate_groups().count(),
        summary
            .duplicate_groups()
            .map(|group| group.files.len())
            .sum::<usize>()
    );

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
        .duplicate_groups()
        .map(|group| {
            let fingerprint = format!("{:016x}", group.fingerprint);
            let max_size = group
                .files
                .iter()
                .map(|file| file.size_bytes)
                .max()
                .unwrap_or(0);
            let keep_index = group
                .files
                .iter()
                .enumerate()
                .filter(|(_, file)| file.size_bytes == max_size)
                .min_by_key(|(_, file)| file.path.to_string_lossy().to_string())
                .map(|(index, _)| index)
                .unwrap_or(0);
            let files = group
                .files
                .iter()
                .enumerate()
                .map(|(index, file)| {
                    let display_name = file
                        .path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or("unknown")
                        .to_string();
                    let info = format_file_info(file);
                    InternalFile {
                        path: file.path.clone(),
                        display_name,
                        info,
                        size_bytes: file.size_bytes,
                        selected: file.size_bytes == max_size && index != keep_index,
                        thumbnail: file.thumbnail.clone(),
                    }
                })
                .collect();

            InternalGroup { fingerprint, files }
        })
        .collect()
}

fn ensure_largest_selected(groups: &mut [InternalGroup]) {
    for group in groups.iter_mut() {
        if group.files.is_empty() {
            continue;
        }
        let max_size = group
            .files
            .iter()
            .map(|file| file.size_bytes)
            .max()
            .unwrap_or(0);
        let keep_index = group
            .files
            .iter()
            .enumerate()
            .find(|(_, file)| file.size_bytes == max_size && !file.selected)
            .map(|(index, _)| index)
            .or_else(|| {
                group
                    .files
                    .iter()
                    .enumerate()
                    .find(|(_, file)| file.size_bytes == max_size)
                    .map(|(index, _)| index)
            })
            .unwrap_or(0);
        for (index, file) in group.files.iter_mut().enumerate() {
            file.selected = file.size_bytes == max_size && index != keep_index;
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
    format!(
        "Groups: {} • Files: {} • Selected: {}",
        group_count, file_count, selected_count
    )
}

fn build_scan_config() -> ScanConfig {
    let mut config = ScanConfig::new(default_extensions(), ThreadingMode::Parallel);
    if let Some(mut dir) = dirs::data_local_dir() {
        dir.push("Camden");
        dir.push("thumbnails");
        config = config.with_thumbnail_root(dir);
    }
    config
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
