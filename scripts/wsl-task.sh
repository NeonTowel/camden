#!/usr/bin/env bash
set -euo pipefail

PWSH=${PWSH:-'/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'}

# Helper script intended to run from WSL. It invokes the Windows MSVC cargo build
# via PowerShell so you can stay inside WSL while producing a Windows executable.

# Determine root of the repository (same logic works both when the script is invoked
# directly or via symlink from WSL).
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

# Convert WSL path (/mnt/c/Users/...) to Windows style (C:\Users\...).
to_windows_path() {
    local path="$1"
    local drive
    local rest
    local rest_win

    drive="$(printf '%s' "$path" | cut -d'/' -f3)"
    rest="${path#/mnt/$drive}"
    rest="${rest#/}"
    rest_win="${rest//\//\\}"

    drive="$(printf '%s' "$drive" | tr '[:lower:]' '[:upper:]')"

    if [ -z "$rest_win" ]; then
        printf "%s:" "$drive"
    else
        printf "%s:\\%s" "$drive" "$rest_win"
    fi
}

win_root="$(to_windows_path "$repo_root")"

# Allow callers to pass additional cargo arguments (e.g., --release).
extra_args="$*"

${PWSH} -ExecutionPolicy Bypass -Command "
    Set-Location -LiteralPath '${win_root}';
    task ${extra_args};
"
