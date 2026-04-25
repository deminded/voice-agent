#!/usr/bin/env bash
# Installs voice-agent systemd units. Requires sudo.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
sudo cp "$HERE/voice-agent-server.service" "$HERE/voice-agent-worker.service" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now voice-agent-server.service voice-agent-worker.service
sudo systemctl status voice-agent-server.service voice-agent-worker.service --no-pager
