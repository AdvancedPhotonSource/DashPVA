#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────
# DashPVA Installer
# ─────────────────────────────────────────────

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
EDITION_FILE="$REPO_DIR/.dashpva_edition"
VENV_DIR="$REPO_DIR/.venv"

# ── Helpers ──────────────────────────────────

info()  { printf "  %s\n" "$*"; }
err()   { printf "  ERROR: %s\n" "$*" >&2; }

print_banner() {
    cat <<'BANNER'

    
     ██████████                     █████      ███████████  █████   █████   █████████  
    ▒▒███▒▒▒▒███                   ▒▒███      ▒▒███▒▒▒▒▒███▒▒███   ▒▒███   ███▒▒▒▒▒███ 
    ▒███   ▒▒███  ██████    █████  ▒███████   ▒███    ▒███ ▒███    ▒███  ▒███    ▒███ 
    ▒███    ▒███ ▒▒▒▒▒███  ███▒▒   ▒███▒▒███  ▒██████████  ▒███    ▒███  ▒███████████ 
    ▒███    ▒███  ███████ ▒▒█████  ▒███ ▒███  ▒███▒▒▒▒▒▒   ▒▒███   ███   ▒███▒▒▒▒▒███ 
    ▒███    ███  ███▒▒███  ▒▒▒▒███ ▒███ ▒███  ▒███          ▒▒▒█████▒    ▒███    ▒███ 
    ██████████  ▒▒████████ ██████  ████ █████ █████           ▒▒███      █████   █████
    ▒▒▒▒▒▒▒▒▒▒    ▒▒▒▒▒▒▒▒ ▒▒▒▒▒▒  ▒▒▒▒ ▒▒▒▒▒ ▒▒▒▒▒             ▒▒▒      ▒▒▒▒▒   ▒▒▒▒▒ 
                                                                                    
                                                                                    
                                                                                    
BANNER
    info "Distributed Analysis and Streaming Hub"
    info "with Process Variable Access"
    echo
}

deactivate_conda() {
    if [[ -z "${CONDA_DEFAULT_ENV:-}" && -z "${CONDA_PREFIX:-}" ]]; then
        return
    fi

    info "Conda environment detected (${CONDA_DEFAULT_ENV:-?}); deactivating for this install."
    info "Your shell session is unaffected — only this script runs without conda."

    # Make `conda deactivate` available in this non-interactive shell
    if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
        local conda_base
        conda_base="$("$CONDA_EXE" info --base 2>/dev/null || true)"
        if [[ -n "$conda_base" && -f "$conda_base/etc/profile.d/conda.sh" ]]; then
            # shellcheck disable=SC1091
            source "$conda_base/etc/profile.d/conda.sh"
        fi
    fi

    # Activations can stack — loop until empty (with a guard)
    local guard=0
    while [[ -n "${CONDA_DEFAULT_ENV:-}" && $guard -lt 10 ]]; do
        if ! conda deactivate 2>/dev/null; then
            break
        fi
        guard=$((guard + 1))
    done

    # Fallback: strip conda from PATH and unset env vars if anything lingers
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        info "Falling back to manual conda cleanup."
        PATH="$(printf '%s' "$PATH" | tr ':' '\n' \
            | grep -vE '/(conda|anaconda3?|miniconda3?|mambaforge)(/|$)' \
            | paste -sd ':' -)"
        export PATH
        unset CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_SHLVL CONDA_PROMPT_MODIFIER CONDA_PYTHON_EXE
    fi
}

ensure_uv() {
    if command -v uv &>/dev/null; then
        info "Found uv: $(command -v uv)"
        return
    fi
    info "uv not found — installing..."
    if command -v curl &>/dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget &>/dev/null; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        err "Neither curl nor wget found. Install uv manually: https://docs.astral.sh/uv/installation/"
        exit 1
    fi
    # Reload PATH so uv is available
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    if ! command -v uv &>/dev/null; then
        err "uv installation succeeded but 'uv' not found on PATH."
        err "Add ~/.local/bin or ~/.cargo/bin to your PATH and re-run."
        exit 1
    fi
    info "uv installed: $(command -v uv)"
}

prompt_edition() {
    echo
    info "Which edition would you like to install?"
    echo
    info "  [1] Full          — streaming + 3D (HKL) + notebooks"
    info "  [2] Area Detector — area detector viewer + live EPICS streaming (lean)"
    info "  [3] Standalone    — post-analysis tools + notebooks (no streaming)"
    echo
    while true; do
        read -rp "  Enter 1, 2, or 3 [default: 1]: " choice
        choice="${choice:-1}"
        case "$choice" in
            1) EDITION="full"; return ;;
            2) EDITION="area-det"; return ;;
            3) EDITION="standalone"; return ;;
            *) info "Please enter 1, 2, or 3." ;;
        esac
    done
}

do_install() {
    local edition="$1"
    info "Installing $edition edition..."
    echo
    case "$edition" in
        full)       uv sync --extra full ;;
        area-det)   uv sync --extra area-det ;;
        standalone) uv sync --extra notebooks ;;
        *)          err "Unknown edition: $edition"; exit 1 ;;
    esac
}

write_edition() {
    echo "$1" > "$EDITION_FILE"
    info "Edition recorded: $1"
}

link_cli() {
    local target="$VENV_DIR/bin/DashPVA"
    local link_dir="$HOME/.local/bin"
    local link_path="$link_dir/DashPVA"

    if [[ ! -x "$target" ]]; then
        err "DashPVA executable not found in .venv — skipping symlink."
        return
    fi

    mkdir -p "$link_dir"
    ln -sf "$target" "$link_path"
    info "Symlinked: $link_path → $target"

    if ! echo "$PATH" | tr ':' '\n' | grep -qx "$link_dir"; then
        echo
        info "NOTE: $link_dir is not on your PATH."
        info "Add this line to your ~/.bashrc or ~/.zshrc:"
        echo
        info "  export PATH=\"\$HOME/.local/bin:\$PATH\""
        echo
    fi
}

print_next_steps() {
    echo
    info "─────────────────────────────────────────────"
    info "Setup complete! Launch DashPVA with:"
    echo
    info "  DashPVA run"
    echo
    info "To update later:"
    echo
    info "  bash install.sh --update"
    info "─────────────────────────────────────────────"
    echo
}

# ── Main ─────────────────────────────────────

main() {
    local edition=""
    local update=false

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --full)       edition="full"; shift ;;
            --area-det)   edition="area-det"; shift ;;
            --standalone) edition="standalone"; shift ;;
            --update)     update=true; shift ;;
            -h|--help)
                echo "Usage: bash install.sh [--full|--area-det|--standalone|--update]"
                exit 0 ;;
            *)
                err "Unknown option: $1"
                exit 1 ;;
        esac
    done

    print_banner

    # Handle --update
    if $update; then
        if [[ -f "$EDITION_FILE" ]]; then
            edition="$(cat "$EDITION_FILE")"
            info "Updating existing $edition installation..."
        else
            info "No previous installation found. Running fresh install."
        fi
    fi

    # Determine edition if not set
    if [[ -z "$edition" ]]; then
        prompt_edition
        edition="$EDITION"
    fi

    # Block downgrade from a Full install to a leaner tier (would leave a
    # mismatched env). Switching requires a fresh environment.
    if [[ -f "$EDITION_FILE" ]]; then
        prev="$(cat "$EDITION_FILE")"
        if [[ "$prev" == "full" && ( "$edition" == "standalone" || "$edition" == "area-det" ) ]]; then
            err "Cannot downgrade from Full to $edition."
            err "To switch, create a fresh environment."
            exit 1
        fi
    fi

    deactivate_conda
    ensure_uv

    cd "$REPO_DIR"
    do_install "$edition"
    write_edition "$edition"
    link_cli
    print_next_steps
}

main "$@"
