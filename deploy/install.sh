#!/usr/bin/env bash
# Installs voice-agent systemd units. Run as root, or as a user with full sudo.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SUDO=""
if [[ $EUID -ne 0 ]]; then
  SUDO="sudo"
fi
$SUDO cp "$HERE/voice-agent-server.service" "$HERE/voice-agent-worker.service" /etc/systemd/system/
$SUDO systemctl daemon-reload
$SUDO systemctl enable --now voice-agent-server.service voice-agent-worker.service
$SUDO systemctl status voice-agent-server.service voice-agent-worker.service --no-pager
