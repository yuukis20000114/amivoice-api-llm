#!/usr/bin/env bash

# -----------------------------------------------------------------------------
# Uithub-like Git Repository Explorer
# -----------------------------------------------------------------------------
# Displays a Git repository's structure and file contents in an AI-friendly format.
#
# Features:
#   - Displays Git-tracked files with line numbers
#   - Shows file structure (up to 2 levels deep)
#   - Highlights media files with GitHub raw URLs
#   - Ignores blank files and large files (>100KB)
#   - Ignores files excluded by .gitignore
#   - Ignores common build, cache, and temporary files
#   - Keeps .md and .json files (only excludes unnecessary files)
#   - Interactive file selection with `fzf` (optional)
#
# Requirements: git, tree, cat, fzf (optional)
#
# Usage:
#   ./show_repo.sh          # Standard output
#   ./show_repo.sh --interactive # Interactive mode (requires fzf)
# -----------------------------------------------------------------------------

LOG_FILE="uithub_output_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -i "$LOG_FILE") 2>&1

set -euo pipefail

# Color codes for output
RED='\033[1;31m'
GREEN='\033[1;32m'
CYAN='\033[1;36m'
YELLOW='\033[1;33m'
RESET='\033[0m'

# Configurable settings
MAX_FILE_SIZE=$((100 * 1024)) # 100KB
TREE_DEPTH=2 # Depth for tree display

# Patterns of files to exclude
EXCLUDE_PATTERNS="\
\.lock$|\
\.o$|\.out$|\.so$|\.a$|\
\.class$|\.jar$|\.war$|\.ear$|\
\.pyc$|\.pyo$|__pycache__/|\
\.beam$|_build/|\
dist/|build/|target/|\.gradle/|\
\.DS_Store$|Thumbs\.db$|\.idea/|\.vscode/|\
node_modules/|bower_components/|\
\.pytest_cache/|\.mypy_cache/|\.cargo/|\
\.log$|\.tmp$|\.swp$"

# Check if inside a Git repository
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo -e "${RED}Error: This is not a Git repository.${RESET}" >&2
  exit 1
fi

# Get GitHub raw content URL
get_github_url() {
  local file="$1"
  local repo_url branch

  repo_url="$(git config --get remote.origin.url | sed -E 's|git@github.com:|https://github.com/|; s|https://github.com/||; s|\.git$||')"
  branch="$(git rev-parse --abbrev-ref HEAD)"

  echo "https://raw.githubusercontent.com/${repo_url}/${branch}/${file}"
}

# Check if a file is binary
is_binary() {
  case "$1" in
    *.png|*.jpg|*.jpeg|*.gif|*.bmp|*.tiff|*.ico|*.svg|*.webp|*.avif|\
    *.mp4|*.mkv|*.mov|*.avi|*.wmv|*.flv|*.webm|*.mpeg|*.mpg|*.m4v|*.3gp)
      return 0 ;;
    *) return 1 ;;
  esac
}

# Check if a file is empty (zero bytes or only whitespace)
is_blank_file() {
  local file="$1"

  # File size is zero? -> Blank
  if [[ ! -s "$file" ]]; then
    return 0
  fi

  # Check if the file contains only whitespace
  if [[ $(grep -cvP '\S' "$file") -eq 0 ]]; then
    return 0
  fi

  return 1
}

# Get list of Git-tracked files, excluding ignored ones and unnecessary files
get_git_tracked_files() {
  git ls-files --exclude-standard -c -o | grep -Ev "${EXCLUDE_PATTERNS}" || true
}

# Display repository structure safely
show_structure() {
  echo -e "${CYAN}📁 Repository Structure (Depth: ${TREE_DEPTH}):${RESET}"

  # Try tree first, fallback to ls if tree fails
  if ! tree -L "${TREE_DEPTH}" 2>/dev/null; then
    echo -e "${YELLOW}⚠ 'tree' command failed, falling back to 'ls -R'${RESET}"
    ls -R | head -n 50  # Avoid massive output
  fi

  echo -e "\n---\n"
}

# Process and display file contents
show_files() {
  local interactive_mode="${1:-false}"

  # Get filtered list of Git-tracked files
  mapfile -t files < <(get_git_tracked_files)

  if [[ "${interactive_mode}" == "true" && -x "$(command -v fzf)" ]]; then
    # Interactive mode: Select files with fzf
    selected_file=$(printf "%s\n" "${files[@]}" | fzf --preview "bat --color=always {}" --height=40%)
    files=("$selected_file")
  fi

  for file in "${files[@]}"; do
    # Skip large files
    if [[ "$(stat -c%s "$file")" -gt "$MAX_FILE_SIZE" ]]; then
      continue
    fi

    # Skip blank files
    if is_blank_file "$file"; then
      continue
    fi

    echo -e "\n${GREEN}==> ${file} <==${RESET}"

    if is_binary "$file"; then
      echo "🔗 $(get_github_url "$file")"
    else
      cat -n "$file"
    fi

    echo -e "\n---"
  done
}

# Show usage/help text
show_help() {
  echo "Usage: $0 [options]"
  echo
  echo "Options:"
  echo "  --interactive   Enable interactive file selection with fzf"
  echo "  --help          Show this help message"
}

# Main function
main() {
  local args=("$@")

  # Parse options
  for arg in "${args[@]}"; do
    case "$arg" in
      --help)
        show_help
        exit 0
        ;;
      --interactive)
        INTERACTIVE_MODE=true
        ;;
      *)
        echo -e "${RED}Unknown option: $arg${RESET}" >&2
        show_help
        exit 1
        ;;
    esac
  done

  # Generate output
  show_structure
  show_files "${INTERACTIVE_MODE:-false}"
}

main "$@"
