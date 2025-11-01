use indicatif::ProgressStyle;

pub fn default_style() -> ProgressStyle {
    match ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
    {
        Ok(style) => style.progress_chars("##-"),
        Err(_) => ProgressStyle::default_bar(),
    }
}
