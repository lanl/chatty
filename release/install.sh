#!/bin/sh
# Per-user Apple Silicon installer. Keep this file usable on its own.
set -eu

install_suite() (
    umask 077
    if [ "$(uname -sm)" != "Darwin arm64" ] || [ "$(sw_vers -productVersion | cut -d. -f1)" -lt 14 ]; then
        echo 'The laptop suite requires macOS 14 or later in a native Apple Silicon terminal.' >&2
        exit 1
    fi

    tag=laptop-2026.11.1
    manifest_sha=e01613235ba76672aaf781f46da134ee26abb604dfaf335fa1acc0c2e1d62837
    uv_sha=5aeca67f2283f75235de40d0bd9c0a5bbc5fce23b6d48b3bf9fa1b2db08382ca
    root="$HOME/Library/Application Support/ChattySuite"
    check_destination() {
        if [ -L "$root/bin" ]; then
            case "$(readlink "$root/bin")" in
                "$root"/runtimes/suite.*/commands) return 0 ;;
            esac
        elif [ ! -e "$root/bin" ]; then
            return 0
        fi
        echo "Cannot activate the suite: $root/bin already exists and is not its runtime link." >&2
        return 1
    }
    check_destination
    mkdir -p "$root/runtimes"
    tmp=$(mktemp -d "$root/.install.XXXXXXXX")
    candidate=
    trap 'rm -rf "$tmp"; if [ -n "$candidate" ]; then rm -rf "$candidate"; fi' EXIT
    trap 'exit 1' HUP INT TERM
    cd "$tmp"

    fetch() {
        curl --proto '=https' --fail --silent --show-error --location --retry 3 --output "$2" "$1"
        printf '%s  %s\n' "$3" "$2" | shasum -a 256 -c -
    }
    field() { plutil -extract "$1" raw -expect string manifest.json; }
    fetch "https://github.com/lanl/chatty/releases/download/$tag/manifest.json" manifest.json "$manifest_sha"
    [ "$(field tag)" = "$tag" ]
    python_version=$(field python)
    lock=platforms.macos-arm64.requirements
    fetch "$(field "$lock.url")" requirements.txt "$(field "$lock.sha256")"

    healthy() {
        "$1/bin/python" -I -m pip --isolated check &&
            "$1/bin/python" -I -c 'from torch import tensor; from faiss import IndexFlatIP; import litkit.cli' || return 1
        for name in chatty text-fetch litkit; do
            version=$("$1/commands/$name" --version) || return 1
            [ "${version##* }" = "$(field "projects.$name.version")" ] || return 1
        done
    }

    active=$(readlink "$root/bin" || true)
    runtime=${active%/commands}
    if [ -n "$runtime" ] && [ -f "$runtime/manifest.sha256" ] &&
        [ "$(cat "$runtime/manifest.sha256")" = "$manifest_sha" ] && healthy "$runtime"; then
        echo "Using installed $tag."
    else
        fetch 'https://github.com/astral-sh/uv/releases/download/0.13.9/uv-aarch64-apple-darwin.tar.gz' uv.tar.gz "$uv_sha"
        tar -xzf uv.tar.gz uv-aarch64-apple-darwin/uv
        candidate=$(mktemp -d "$root/runtimes/suite.XXXXXXXX")
        export UV_PYTHON_INSTALL_DIR="$root/python"
        # uv optionally runs install_name_tool for extension builds. On a stock Mac
        # that executable is a shim that opens the Xcode installation dialog.
        PATH="$tmp/uv-aarch64-apple-darwin" ./uv-aarch64-apple-darwin/uv venv --quiet \
            --no-config --no-project --managed-python --python "$python_version" "$candidate"
        py="$candidate/bin/python"
        "$py" -I -m ensurepip
        "$py" -I -m pip --isolated --disable-pip-version-check install --no-index --only-binary=:all: --require-hashes \
            --report "$candidate/pip-report.json" -r requirements.txt
        mkdir "$candidate/commands"
        for name in chatty text-fetch litkit; do
            ln -s "../bin/$name" "$candidate/commands/$name"
        done
        healthy "$candidate"
        echo "$manifest_sha" > "$candidate/manifest.sha256"
        ln -s "$candidate/commands" next
        # Replace the link atomically. Keep cleanup armed until the switch succeeds.
        check_destination
        trap '' HUP INT TERM
        "$py" -I -c 'import os, sys; os.replace("next", sys.argv[1])' "$root/bin" </dev/null
        candidate=
        trap 'exit 1' HUP INT TERM
    fi

    add_path() {
        if ! grep -Fqx '# Chatty laptop suite commands' "$1" 2>/dev/null; then
            cat >> "$1" <<'PROFILE'

# Chatty laptop suite commands
case ":$PATH:" in
    *":$HOME/Library/Application Support/ChattySuite/bin:"*) ;;
    *) export PATH="$HOME/Library/Application Support/ChattySuite/bin:$PATH" ;;
esac
PROFILE
        fi
    }
    # A terminal can start either shell regardless of the account's login shell.
    add_path "${ZDOTDIR:-$HOME}/.zshrc"
    profile="$HOME/.bash_profile"
    for name in .bash_profile .bash_login .profile; do
        if [ -f "$HOME/$name" ] && [ -r "$HOME/$name" ]; then
            profile="$HOME/$name"
            break
        fi
    done
    add_path "$profile"
    add_path "$HOME/.bashrc"
    echo "Installed $tag. Open a new terminal and run chatty --version."
)

install_suite
