#!/usr/bin/env bash
# Installs nginx config + Let's Encrypt cert for va.noomarxism.ru.
# Run as root: sudo bash deploy/install_nginx.sh
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "must run as root: sudo $0" >&2
  exit 1
fi

HERE="$(cd "$(dirname "$0")" && pwd)"
DOMAIN="va.noomarxism.ru"

# 1. Place nginx config
cp "$HERE/$DOMAIN.conf" "/etc/nginx/sites-available/$DOMAIN"
ln -sf "/etc/nginx/sites-available/$DOMAIN" "/etc/nginx/sites-enabled/$DOMAIN"

# 2. Issue cert via certbot. The --nginx plugin patches nginx temporarily for HTTP-01.
# If certbot isn't installed: apt install -y certbot python3-certbot-nginx
certbot --nginx --non-interactive --agree-tos --redirect \
  -d "$DOMAIN" --email demik.open@gmail.com \
  --no-eff-email

# 3. Final nginx test + reload (certbot already reloaded; this is belt-and-suspenders)
nginx -t
systemctl reload nginx

echo
echo "Done. Verify with:"
echo "  curl -I https://$DOMAIN/health"
echo "  curl -I 'https://$DOMAIN/lk?key=<TOKEN_FROM_ENV>'"
