#!/usr/bin/env bash
# Installs nginx config + Let's Encrypt cert for va.noomarxism.ru.
# Run as root: bash deploy/install_nginx.sh
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "must run as root: su - then bash $0" >&2
  exit 1
fi

HERE="$(cd "$(dirname "$0")" && pwd)"
DOMAIN="va.noomarxism.ru"
CONF_AVAILABLE="/etc/nginx/sites-available/$DOMAIN"
CONF_ENABLED="/etc/nginx/sites-enabled/$DOMAIN"

# 1. Drop any prior broken state from a failed earlier run.
rm -f "$CONF_AVAILABLE" "$CONF_ENABLED"

# 2. Place fresh HTTP-only config.
cp "$HERE/$DOMAIN.conf" "$CONF_AVAILABLE"
ln -sf "$CONF_AVAILABLE" "$CONF_ENABLED"

# 3. Validate + reload so nginx is serving HTTP for va.noomarxism.ru.
nginx -t
systemctl reload nginx

# 4. Issue cert via certbot --nginx. It edits the config in-place to add
#    a 443 SSL server block + HTTP→HTTPS redirect, then reloads nginx.
certbot --nginx --non-interactive --agree-tos --redirect \
  -d "$DOMAIN" --email demik.open@gmail.com --no-eff-email

# 5. Final sanity check.
nginx -t
systemctl reload nginx

echo
echo "Done. Verify with:"
echo "  curl -I https://$DOMAIN/health"
echo "  cat /tmp/voice-agent-url.txt   # full URL with token"
