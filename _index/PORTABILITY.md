# Moving the Art Collection to another Mac

The whole system lives inside this one folder (`Art Collection/`) — images, the
index (`_index/index.db`), the research corpus, and all scripts. It is now
**location-independent**: the scripts find the collection root from their own
path, so it works at any folder / any username. The only things that don't travel
are the Python venv (machine-specific) and the login service (lives in
`~/Library/`); both are rebuilt by the two setup scripts below.

## Move it

1. **Copy the folder** to the new Mac. Skip the venv and caches — they get rebuilt:
   ```bash
   rsync -avh --progress \
     --exclude '_index/.venv' \
     --exclude '_index/__pycache__' \
     --exclude '.DS_Store' \
     "/Volumes/OldDrive/Art Collection/" \
     "$HOME/Documents/Art Collection/"
   ```
   (Or drag the folder in Finder — just delete `_index/.venv` afterward to save 1.1 GB
   of dead weight; `setup.sh` rebuilds it correctly.)

2. **Rebuild the venv** (needs Homebrew Python 3.12: `brew install python@3.12`):
   ```bash
   bash "$HOME/Documents/Art Collection/_index/setup.sh"
   ```

3. **Turn on the always-on gallery + aliases:**
   ```bash
   bash "$HOME/Documents/Art Collection/_index/install-service.sh"
   ```
   Then open a new terminal (or `source ~/.zshrc`). Gallery is at
   http://localhost:8756 and restarts itself at login.

That's it. `index.db` and every dossier/tag came along in the copy — no
re-embedding, no re-OCR, no re-describing.

## Notes
- Re-running `install-service.sh` after a later move re-points the service and
  aliases at the new path automatically (it strips and rewrites its own block).
- `ocrmac` (OCR) uses Apple Vision, so this is macOS-only by design — which is fine,
  it only ever runs on your Macs.
- To uninstall the service:
  `launchctl bootout gui/$(id -u)/com.foggy.artserve && rm ~/Library/LaunchAgents/com.foggy.artserve.plist`
