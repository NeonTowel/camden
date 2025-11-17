mod cli;

use camden_core::{
    count_entries, move_duplicates, print_duplicates, progress, scan, write_json, ScanConfig,
};
use cli::CliConfig;
use indicatif::ProgressBar;
use std::fs;
use std::path::Path;
use std::sync::Arc;

fn main() {
    let config = CliConfig::from_env().unwrap_or_else(|err| {
        eprintln!("{}", err);
        std::process::exit(1);
    });

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
