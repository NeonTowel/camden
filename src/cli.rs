use std::env;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::path::PathBuf;

pub struct CliConfig {
    pub root: PathBuf,
    pub target: Option<PathBuf>,
    pub threading: ThreadingMode,
    pub extensions: Vec<String>,
}

#[derive(Clone, Copy)]
pub enum ThreadingMode {
    Parallel,
    Sequential,
}

#[derive(Debug)]
pub enum CliError {
    MissingRoot,
    InvalidFlag(String),
}

impl CliConfig {
    pub fn from_env() -> Result<Self, CliError> {
        let mut root: Option<PathBuf> = None;
        let mut target: Option<PathBuf> = None;
        let mut threading = ThreadingMode::Parallel;

        for arg in env::args().skip(1) {
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
