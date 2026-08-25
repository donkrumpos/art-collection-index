#!/usr/bin/env bash
# Install the always-on gallery (a launchd LaunchAgent) + shell aliases on THIS Mac.
# Paths are detected at run time, so this works at any location / username.
# Run setup.sh first (needs the venv). Usage:  bash _index/install-service.sh [port]
set -euo pipefail

INDEX="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYBIN="$INDEX/.venv/bin/python"
PORT="${1:-8756}"
LABEL="com.foggy.artserve"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

[ -x "$PYBIN" ] || { echo "No venv at $INDEX/.venv — run:  bash \"$INDEX/setup.sh\"" >&2; exit 1; }
mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"

# --- LaunchAgent (auto-start at login, auto-restart on crash) ---
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>$LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>$PYBIN</string>
        <string>$INDEX/serve.py</string>
        <string>--port</string>
        <string>$PORT</string>
    </array>
    <key>WorkingDirectory</key><string>$INDEX</string>
    <key>EnvironmentVariables</key>
    <dict><key>PATH</key><string>/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string></dict>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>StandardOutPath</key><string>$HOME/Library/Logs/artserve.log</string>
    <key>StandardErrorPath</key><string>$HOME/Library/Logs/artserve.log</string>
</dict>
</plist>
EOF

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "Gallery service running -> http://localhost:$PORT  (auto-starts at login)"

# --- Shell aliases (idempotent block in ~/.zshrc) ---
RC="$HOME/.zshrc"
MARK="# >>> art-collection aliases >>>"
END="# <<< art-collection aliases <<<"
if [ -f "$RC" ] && grep -qF "$MARK" "$RC"; then
  # strip the old block so paths stay correct after a move
  /usr/bin/sed -i '' "/$MARK/,/$END/d" "$RC"
fi
cat >> "$RC" <<EOF
$MARK
alias artsearch='"$PYBIN" "$INDEX/search.py"'
alias artserve='"$PYBIN" "$INDEX/serve.py"'
alias artindex='"$PYBIN" "$INDEX/embed.py"'
alias arttags='"$PYBIN" "$INDEX/seed_tags.py"'
alias artread='"$PYBIN" "$INDEX/research_search.py"'
alias artmean='"$PYBIN" "$INDEX/research_semantic.py"'
$END
EOF
echo "Aliases written to ~/.zshrc (open a new terminal, or: source ~/.zshrc)"
