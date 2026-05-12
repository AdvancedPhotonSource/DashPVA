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
    info "  [1] Full        — live streaming + pvaccess/EPICS + bayesian"
    info "  [2] Standalone  — post-analysis tools only"
    echo
    while true; do
        read -rp "  Enter 1 or 2 [default: 1]: " choice
        choice="${choice:-1}"
        case "$choice" in
            1) EDITION="full"; return ;;
            2) EDITION="standalone"; return ;;
            *) info "Please enter 1 or 2." ;;
        esac
    done
}

do_install() {
    local edition="$1"
    info "Installing $edition edition..."
    echo
    if [[ "$edition" == "full" ]]; then
        uv sync --extra full
    else
        uv sync
    fi
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
            --standalone) edition="standalone"; shift ;;
            --update)     update=true; shift ;;
            -h|--help)
                echo "Usage: bash install.sh [--full|--standalone|--update]"
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

    # Block downgrade
    if [[ "$edition" == "standalone" && -f "$EDITION_FILE" ]]; then
        prev="$(cat "$EDITION_FILE")"
        if [[ "$prev" == "full" ]]; then
            err "Cannot downgrade from Full to Standalone."
            err "To switch, create a fresh environment."
            exit 1
        fi
    fi

    ensure_uv

    cd "$REPO_DIR"
    do_install "$edition"
    write_edition "$edition"
    link_cli
    print_next_steps
}

main "$@"
