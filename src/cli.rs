use camden_core::ThreadingMode;
use std::env;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::path::PathBuf;

#[derive(Debug, PartialEq, Eq)]
pub struct CliConfig {
    pub root: PathBuf,
    pub target: Option<PathBuf>,
    pub threading: ThreadingMode,
    pub extensions: Vec<String>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum CliError {
    MissingRoot,
    InvalidFlag(String),
}

impl CliConfig {
    pub fn from_env() -> Result<Self, CliError> {
        Self::from_iter(env::args().skip(1))
    }

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

impl Display for CliError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingRoot => write!(f, "root directory argument is required"),
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
    fn parses_root_only() {
        let config = CliConfig::from_iter(vec![String::from("./images")]).unwrap();
        assert_eq!(config.root, PathBuf::from("./images"));
        assert!(config.target.is_none());
        assert_eq!(config.threading, ThreadingMode::Parallel);
        assert_eq!(config.extensions, default_extensions());
    }

    #[test]
    fn parses_positional_target() {
        let config =
            CliConfig::from_iter(vec![String::from("./images"), String::from("./duplicates")])
                .unwrap();
        assert_eq!(config.root, PathBuf::from("./images"));
        assert_eq!(config.target, Some(PathBuf::from("./duplicates")));
    }

    #[test]
    fn parses_flag_arguments() {
        let config = CliConfig::from_iter(vec![
            String::from("--root=./images"),
            String::from("--target=./duplicates"),
            String::from("--no-thread"),
        ])
        .unwrap();
        assert_eq!(config.root, PathBuf::from("./images"));
        assert_eq!(config.target, Some(PathBuf::from("./duplicates")));
        assert_eq!(config.threading, ThreadingMode::Sequential);
    }

    #[test]
    fn returns_error_for_missing_root() {
        let result = CliConfig::from_iter(Vec::<String>::new());
        assert!(matches!(result, Err(CliError::MissingRoot)));
    }

    #[test]
    fn returns_error_for_invalid_flag() {
        let result = CliConfig::from_iter(vec![
            String::from("./images"),
            String::from("--unsupported"),
        ]);
        match result {
            Err(CliError::InvalidFlag(flag)) => assert_eq!(flag, "--unsupported"),
            other => panic!("unexpected result: {:?}", other),
        }
    }

    #[test]
    fn returns_error_for_excess_arguments() {
        let result = CliConfig::from_iter(vec![
            String::from("./images"),
            String::from("./duplicates"),
            String::from("./extra"),
        ]);
        assert!(matches!(result, Err(CliError::InvalidFlag(extra)) if extra == "./extra"));
    }
}
