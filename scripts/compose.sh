#!/bin/sh
set -eu

cd "$(dirname "$0")/.."

if sudo docker compose version >/dev/null 2>&1; then
    sudo docker compose "$@"
elif command -v docker-compose >/dev/null 2>&1; then
    sudo docker-compose "$@"
else
    echo "Docker Compose is not installed." >&2
    exit 1
fi
