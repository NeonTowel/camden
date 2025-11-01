mod cli;
mod operations;
mod progress;
mod reporting;
mod scanner;

use cli::CliConfig;
use indicatif::ProgressBar;
use operations::move_duplicates;
use reporting::{print_duplicates, write_json};
use scanner::{count_entries, scan};
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

    let checksum_map = scan(
        &config.root,
        &config.extensions,
        config.threading,
        &progress_bar,
    );
    progress_bar.finish_with_message("Scan complete");

    print_duplicates(&checksum_map);
    match write_json(&checksum_map, Path::new("identical_files.json")) {
        Ok(_) => println!("JSON output written to identical_files.json"),
        Err(error) => eprintln!("Error writing JSON output: {}", error),
    }

    if let Some(target_dir) = config.target.as_ref() {
        if !target_dir.exists() {
            fs::create_dir_all(target_dir).expect("Failed to create target directory");
        }
        match move_duplicates(&checksum_map, target_dir) {
            Ok(stats) => println!(
                "Duplicate files moved to {} ({})",
                target_dir.display(),
                stats.moved
            ),
            Err(error) => eprintln!("Error moving duplicate files: {}", error),
        }
    }
}
