#!/bin/bash
set -e

echo "=========================================="
echo "  Enigma API - SSL Certificate Setup"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check arguments
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <domain> <email>"
    echo "Example: $0 api.example.com admin@example.com"
    exit 1
fi

DOMAIN=$1
EMAIL=$2
APP_DIR="/home/ubuntu/app"

echo ""
echo "Domain: $DOMAIN"
echo "Email: $EMAIL"
echo ""

# Check if running from app directory
if [ ! -f "$APP_DIR/docker-compose.prod.yml" ]; then
    print_error "docker-compose.prod.yml not found in $APP_DIR"
    print_error "Please run this script from the app directory"
    exit 1
fi

cd $APP_DIR

# Ensure nginx is running
echo "Ensuring nginx is running..."
docker compose -f docker-compose.prod.yml up -d nginx
sleep 5
print_status "Nginx is running"

# Request certificate
echo ""
echo "Requesting SSL certificate..."
docker compose -f docker-compose.prod.yml run --rm certbot certonly \
    --webroot \
    --webroot-path=/var/www/certbot \
    --email $EMAIL \
    --agree-tos \
    --no-eff-email \
    -d $DOMAIN

if [ $? -eq 0 ]; then
    print_status "SSL certificate obtained successfully"
else
    print_error "Failed to obtain SSL certificate"
    exit 1
fi

# Create SSL nginx config from template
echo ""
echo "Configuring nginx for SSL..."
SSL_CONF="$APP_DIR/infra/nginx/conf.d/ssl.conf"

if [ -f "$APP_DIR/infra/nginx/conf.d/ssl.conf.template" ]; then
    cp "$APP_DIR/infra/nginx/conf.d/ssl.conf.template" "$SSL_CONF"
    sed -i "s/YOUR_DOMAIN/$DOMAIN/g" "$SSL_CONF"
    print_status "SSL nginx config created"
else
    print_warning "ssl.conf.template not found, creating default config"
    cat > "$SSL_CONF" << EOF
server {
    listen 80;
    listen [::]:80;
    server_name $DOMAIN;

    location /.well-known/acme-challenge/ {
        root /var/www/certbot;
    }

    location / {
        return 301 https://\$host\$request_uri;
    }
}

server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name $DOMAIN;

    ssl_certificate /etc/letsencrypt/live/$DOMAIN/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/$DOMAIN/privkey.pem;

    ssl_session_timeout 1d;
    ssl_session_cache shared:SSL:50m;
    ssl_session_tickets off;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:DHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;

    add_header Strict-Transport-Security "max-age=63072000" always;

    location /api {
        proxy_pass http://app:3001;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_connect_timeout 60s;
        proxy_send_timeout 120s;
        proxy_read_timeout 120s;
        client_max_body_size 50M;
    }

    location /assets {
        proxy_pass http://app:3001;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
    }

    location = / {
        return 302 /api/docs;
    }

    location = /api/health {
        proxy_pass http://app:3001;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
    }
}
EOF
    print_status "SSL nginx config created"
fi

# Remove default.conf to avoid conflicts (HTTP redirect is in ssl.conf)
if [ -f "$APP_DIR/infra/nginx/conf.d/default.conf" ]; then
    mv "$APP_DIR/infra/nginx/conf.d/default.conf" "$APP_DIR/infra/nginx/conf.d/default.conf.bak"
    print_status "Backed up default.conf"
fi

# Reload nginx
echo ""
echo "Reloading nginx..."
docker compose -f docker-compose.prod.yml exec nginx nginx -s reload
print_status "Nginx reloaded"

# Test HTTPS
echo ""
echo "Testing HTTPS connection..."
sleep 2
if curl -s -o /dev/null -w "%{http_code}" "https://$DOMAIN/api/health" | grep -q "200"; then
    print_status "HTTPS is working!"
else
    print_warning "HTTPS test returned non-200 status. Please check manually."
fi

echo ""
echo "=========================================="
echo "  SSL Setup Complete!"
echo "=========================================="
echo ""
echo "Your API is now accessible at:"
echo "  https://$DOMAIN/api"
echo "  https://$DOMAIN/api/docs"
echo ""
echo "Certificate will auto-renew via certbot container."
echo ""
print_status "SSL setup completed successfully"
