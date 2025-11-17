mod cli;

use camden_core::{
    count_entries, create_snapshot, move_duplicates, print_duplicates, progress, scan, write_json,
    write_snapshot, ScanConfig,
};
use cli::{CliConfig, Command, PreviewConfig};
use indicatif::ProgressBar;
use std::fs;
use std::path::Path;
use std::sync::Arc;

fn main() {
    let command = Command::from_env().unwrap_or_else(|err| {
        eprintln!("{}", err);
        std::process::exit(1);
    });

    match command {
        Command::Scan(config) => run_scan(config),
        Command::Preview(config) => run_preview(config),
    }
}

fn run_scan(config: CliConfig) {
    let total_files = count_entries(&config.root);
    let progress_bar = ProgressBar::new(total_files);
    let progress_bar = Arc::new(progress_bar);

    progress_bar.set_style(progress::default_style());

    let scan_config = ScanConfig::new(config.extensions.clone(), config.threading);
    let summary = scan(&config.root, &scan_config, &progress_bar);
    progress_bar.finish_with_message("Scan complete");

    print_duplicates(&summary);
    match write_json(&summary, Path::new("identical_files.json")) {
        Ok(_) => println!("JSON output written to identical_files.json"),
        Err(error) => eprintln!("Error writing JSON output: {}", error),
    }

    if let Some(target_dir) = config.target.as_ref() {
        if !target_dir.exists() {
            fs::create_dir_all(target_dir).expect("Failed to create target directory");
        }
        match move_duplicates(&summary, target_dir) {
            Ok(stats) => println!(
                "Duplicate files moved to {} ({})",
                target_dir.display(),
                stats.moved
            ),
            Err(error) => eprintln!("Error moving duplicate files: {}", error),
        }
    }
}

fn run_preview(config: PreviewConfig) {
    let total_files = count_entries(&config.root);
    let progress_bar = ProgressBar::new(total_files);
    let progress_bar = Arc::new(progress_bar);
    progress_bar.set_style(progress::default_style());

    let mut scan_config = ScanConfig::new(config.extensions.clone(), config.threading);
    if let Some(root) = config.thumbnail_root() {
        scan_config = scan_config.with_thumbnail_root(root.to_path_buf());
    }

    let summary = scan(&config.root, &scan_config, &progress_bar);
    progress_bar.finish_with_message("Preview scan complete");

    let snapshot = create_snapshot(&config.root, summary.clone());
    match write_snapshot(&snapshot, &config.output) {
        Ok(_) => println!("Preview snapshot written to {}", config.output.display()),
        Err(error) => eprintln!(
            "Error writing preview snapshot to {}: {}",
            config.output.display(),
            error
        ),
    }

    print_duplicates(&summary);
}
