use camden_core::ThreadingMode;
use std::env;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::path::{Path, PathBuf};

#[derive(Debug, PartialEq, Eq)]
pub enum Command {
    Scan(CliConfig),
    Preview(PreviewConfig),
}

#[derive(Debug, PartialEq, Eq)]
pub struct CliConfig {
    pub root: PathBuf,
    pub target: Option<PathBuf>,
    pub threading: ThreadingMode,
    pub extensions: Vec<String>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct PreviewConfig {
    pub root: PathBuf,
    pub output: PathBuf,
    pub threading: ThreadingMode,
    pub extensions: Vec<String>,
    pub thumbnail_root: Option<PathBuf>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum CliError {
    MissingRoot,
    MissingOutput,
    InvalidFlag(String),
}

impl Command {
    pub fn from_env() -> Result<Self, CliError> {
        Self::from_iter(env::args().skip(1))
    }

    pub fn from_iter<I>(args: I) -> Result<Self, CliError>
    where
        I: IntoIterator<Item = String>,
    {
        let mut args = args.into_iter();
        match args.next() {
            Some(first) if first == "preview-scan" => {
                PreviewConfig::parse(args).map(Command::Preview)
            }
            Some(first) => {
                let mut rest = vec![first];
                rest.extend(args);
                CliConfig::from_iter(rest.into_iter()).map(Command::Scan)
            }
            None => Err(CliError::MissingRoot),
        }
    }
}

impl CliConfig {
    pub fn from_iter<I>(args: I) -> Result<Self, CliError>
    where
        I: IntoIterator<Item = String>,
    {
        Self::parse(args.into_iter())
    }

    fn parse<I>(mut args: I) -> Result<Self, CliError>
    where
        I: Iterator<Item = String>,
    {
        let mut root: Option<PathBuf> = None;
        let mut target: Option<PathBuf> = None;
        let mut threading = ThreadingMode::Parallel;

        for arg in args.by_ref() {
            if arg.starts_with("--") {
                if arg == "--no-thread" {
                    threading = ThreadingMode::Sequential;
                    continue;
                }
                if let Some(value) = arg.strip_prefix("--target=") {
                    target = Some(PathBuf::from(value));
                    continue;
                }
                if let Some(value) = arg.strip_prefix("--root=") {
                    root = Some(PathBuf::from(value));
                    continue;
                }
                return Err(CliError::InvalidFlag(arg));
            }

            if root.is_none() {
                root = Some(PathBuf::from(&arg));
                continue;
            }

            if target.is_none() {
                target = Some(PathBuf::from(&arg));
                continue;
            }

            return Err(CliError::InvalidFlag(arg));
        }

        let root = root.ok_or(CliError::MissingRoot)?;

        Ok(Self {
            root,
            target,
            threading,
            extensions: default_extensions(),
        })
    }
}

impl PreviewConfig {
    fn parse<I>(mut args: I) -> Result<Self, CliError>
    where
        I: Iterator<Item = String>,
    {
        let mut root: Option<PathBuf> = None;
        let mut output: Option<PathBuf> = None;
        let mut thumbnail_root: Option<PathBuf> = None;
        let mut threading = ThreadingMode::Parallel;

        for arg in args.by_ref() {
            if arg.starts_with("--") {
                if arg == "--no-thread" {
                    threading = ThreadingMode::Sequential;
                    continue;
                }
                if let Some(value) = arg.strip_prefix("--root=") {
                    root = Some(PathBuf::from(value));
                    continue;
                }
                if let Some(value) = arg.strip_prefix("--output=") {
                    output = Some(PathBuf::from(value));
                    continue;
                }
                if let Some(value) = arg.strip_prefix("--thumbnail-root=") {
                    thumbnail_root = Some(PathBuf::from(value));
                    continue;
                }
                return Err(CliError::InvalidFlag(arg));
            }

            if root.is_none() {
                root = Some(PathBuf::from(&arg));
                continue;
            }

            if output.is_none() {
                output = Some(PathBuf::from(&arg));
                continue;
            }

            return Err(CliError::InvalidFlag(arg));
        }

        let root = root.ok_or(CliError::MissingRoot)?;
        let output = output
            .or_else(|| camden_core::default_snapshot_path())
            .ok_or(CliError::MissingOutput)?;

        Ok(Self {
            root,
            output,
            threading,
            extensions: default_extensions(),
            thumbnail_root,
        })
    }

    pub fn thumbnail_root(&self) -> Option<&Path> {
        self.thumbnail_root.as_deref()
    }
}

impl Display for CliError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingRoot => write!(f, "root directory argument is required"),
            Self::MissingOutput => write!(f, "snapshot output path is required"),
            Self::InvalidFlag(flag) => write!(f, "unrecognized argument: {}", flag),
        }
    }
}

impl Error for CliError {}

fn default_extensions() -> Vec<String> {
    vec![
        "jpg".to_string(),
        "jpeg".to_string(),
        "png".to_string(),
        "gif".to_string(),
        "bmp".to_string(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_scan_root_only() {
        let command = Command::from_iter(vec![String::from("./images")]).unwrap();
        match command {
            Command::Scan(config) => {
                assert_eq!(config.root, PathBuf::from("./images"));
                assert!(config.target.is_none());
                assert_eq!(config.threading, ThreadingMode::Parallel);
                assert_eq!(config.extensions, default_extensions());
            }
            _ => panic!("expected scan command"),
        }
    }

    #[test]
    fn parses_scan_flags() {
        let command = Command::from_iter(vec![
            String::from("--root=./images"),
            String::from("--target=./duplicates"),
            String::from("--no-thread"),
        ])
        .unwrap();
        match command {
            Command::Scan(config) => {
                assert_eq!(config.root, PathBuf::from("./images"));
                assert_eq!(config.target, Some(PathBuf::from("./duplicates")));
                assert_eq!(config.threading, ThreadingMode::Sequential);
            }
            _ => panic!("expected scan command"),
        }
    }

    #[test]
    fn parses_preview_with_flags() {
        let command = Command::from_iter(vec![
            String::from("preview-scan"),
            String::from("--root=./images"),
            String::from("--output=./cache/groups.json"),
            String::from("--thumbnail-root=./thumbs"),
            String::from("--no-thread"),
        ])
        .unwrap();
        match command {
            Command::Preview(config) => {
                assert_eq!(config.root, PathBuf::from("./images"));
                assert_eq!(config.output, PathBuf::from("./cache/groups.json"));
                assert_eq!(config.thumbnail_root, Some(PathBuf::from("./thumbs")));
                assert_eq!(config.threading, ThreadingMode::Sequential);
            }
            _ => panic!("expected preview command"),
        }
    }

    #[test]
    fn preview_requires_root() {
        let result = Command::from_iter(vec![String::from("preview-scan")]);
        assert!(matches!(result, Err(CliError::MissingRoot)));
    }
}
