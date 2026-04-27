#!/usr/bin/env bash
# Install nginx reverse-proxy + Let's Encrypt cert for the voice-agent web UI.
# Run as root: VOICE_DOMAIN=voice.example.com VOICE_EMAIL=you@example.com \
#              bash deploy/install_nginx.sh
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "must run as root: su - then bash $0" >&2
  exit 1
fi

: "${VOICE_DOMAIN:?Set VOICE_DOMAIN to the public hostname for /lk}"
: "${VOICE_EMAIL:?Set VOICE_EMAIL for Let's Encrypt registration}"

HERE="$(cd "$(dirname "$0")" && pwd)"
DOMAIN="$VOICE_DOMAIN"
CONF_AVAILABLE="/etc/nginx/sites-available/$DOMAIN"
CONF_ENABLED="/etc/nginx/sites-enabled/$DOMAIN"

# 1. Drop any prior broken state from a failed earlier run.
rm -f "$CONF_AVAILABLE" "$CONF_ENABLED"

# 2. Place fresh HTTP-only config, substituting the chosen hostname.
sed "s/voice\.example\.com/$DOMAIN/g" "$HERE/voice-agent.conf.example" > "$CONF_AVAILABLE"
ln -sf "$CONF_AVAILABLE" "$CONF_ENABLED"

# 3. Validate + reload so nginx is serving HTTP for the chosen hostname.
nginx -t
systemctl reload nginx

# 4. Issue cert via certbot --nginx. It edits the config in-place to add
#    a 443 SSL server block + HTTP→HTTPS redirect, then reloads nginx.
certbot --nginx --non-interactive --agree-tos --redirect \
  -d "$DOMAIN" --email "$VOICE_EMAIL" --no-eff-email

# 5. Final sanity check.
nginx -t
systemctl reload nginx

echo
echo "Done. Verify with:"
echo "  curl -I https://$DOMAIN/health"
