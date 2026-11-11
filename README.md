# Starting the project’s Docker containers

These steps assume you have saved all five file groups, applied the revised Dockerfile and NUL-character fix, and installed Docker.

Run commands from your **normal user account**, not a root shell. Your `scripts/compose.sh` wrapper either uses `sudo` internally or uses your account’s Docker-group access, depending on which version you saved.

## 1. Open a terminal and enter the project directory

```bash
cd "$HOME/localrag"
pwd
ls -la
```

The printed directory should end in `/localrag`. You should see:

```text
Dockerfile
compose.yaml
compose.gpu.yaml
.env
manage.py
requirements.txt
config/
library/
monitoring/
scripts/
templates/
theme/
```

Check that the Compose wrapper is executable:

```bash
chmod +x scripts/compose.sh
```

## 2. Start Docker itself

Docker’s background service must be running before containers can start.

```bash
sudo systemctl start docker
sudo systemctl is-active docker
```

The second command should print:

```text
active
```

To have Docker start automatically when the machine boots:

```bash
sudo systemctl enable docker
```

## 3. Validate the project configuration

```bash
cd "$HOME/localrag"

./scripts/compose.sh config --quiet
```

No output usually means configuration validation succeeded. If an error appears, resolve it before continuing.

**Do not print or share your `.env` file—it contains credentials.**

## 4. Choose CPU or GPU startup

Use **one** of the following paths.

### Option A: CPU startup

This requires no NVIDIA container configuration.

Build the application images:

```bash
./scripts/compose.sh build web worker
```

Start every service in the background:

```bash
./scripts/compose.sh up -d
```

### Option B: NVIDIA GPU startup

Use this only if you have already configured the NVIDIA driver and Container Toolkit.

First check the host GPU:

```bash
nvidia-smi
```

If that fails, use CPU startup until the host driver is fixed.

Build the application images:

```bash
./scripts/compose.sh \
  -f compose.yaml \
  -f compose.gpu.yaml \
  build web worker
```

Start every service with the GPU override:

```bash
./scripts/compose.sh \
  -f compose.yaml \
  -f compose.gpu.yaml \
  up -d
```

**The override adds GPU access to the existing Ollama service; it does not create a second Ollama service.**

For subsequent GPU startup, shutdown, or recreation commands, continue including both `-f` arguments.

## 5. Check the container states

For CPU deployment:

```bash
./scripts/compose.sh ps -a
```

For GPU deployment:

```bash
./scripts/compose.sh \
  -f compose.yaml \
  -f compose.gpu.yaml \
  ps -a
```

You should see these six services:

| Service | Purpose |
|---|---|
| `db` | Postgres metadata, passages, and vectors |
| `ollama` | Local model runtime |
| `web` | Django interface |
| `worker` | Background document ingestion |
| `prometheus` | Metrics collection |
| `grafana` | Monitoring dashboards |

On startup, Postgres initializes first. The web and worker services wait for its health check; the web service then initializes application tables.

If a container is `Exited` or repeatedly `Restarting`, inspect its logs.

## 6. Inspect startup logs

These commands work for either deployment because they only read logs:

```bash
./scripts/compose.sh logs --tail=100 db
./scripts/compose.sh logs --tail=100 web worker
./scripts/compose.sh logs --tail=100 ollama prometheus grafana
```

To follow the application logs continuously:

```bash
./scripts/compose.sh logs -f web worker
```

Press **Ctrl+C** to stop following logs. That does **not** stop the containers.

## 7. Download models on the first startup

The intended setup is **one answer model and one embedding model**, both stored by the same Ollama container.

```bash
./scripts/compose.sh exec -T ollama \
  ollama pull llama3.1:8b

./scripts/compose.sh exec -T ollama \
  ollama pull nomic-embed-text
```

Check the installed models:

```bash
./scripts/compose.sh exec -T ollama ollama list
```

If you already downloaded them, you do not need to repeat these pulls for ordinary startup. The `models` volume persists across container restarts.

If you deliberately changed `GENERATION_MODEL` in `.env` to `llama3.2:3b`, pull that model instead of the 8B model.

## 8. Verify Django and the interface

Run Django’s configuration checks:

```bash
./scripts/compose.sh exec -T web \
  python manage.py check
```

Check that the interface responds:

```bash
curl --fail \
  --retry 20 \
  --retry-delay 3 \
  --retry-connrefused \
  http://127.0.0.1:1337/ >/dev/null
```

Check the compiled stylesheet:

```bash
curl --fail \
  http://127.0.0.1:1337/static/css/dist/styles.css \
  >/dev/null
```

Check metrics:

```bash
curl --fail http://127.0.0.1:1337/metrics/
```

These are startup checks—not proof that extraction, retrieval, or generated answers meet the acceptance criteria.

## 9. Open the application

In your browser, open:

- **Application:** `http://127.0.0.1:1337/`
- **Grafana:** `http://127.0.0.1:3000/`

Use **Library** to upload documents and **Index status** to monitor ingestion.

For a fresh Grafana volume, the initial login supplied by this deployment is `admin` / `admin`. Set a new password when prompted. If you previously changed it, use that password.

For GPU deployment, after generating an answer, check model placement:

```bash
./scripts/compose.sh exec -T ollama ollama ps
```

---

# Everyday commands

## Start again after a reboot

You normally **do not rebuild or download models again**.

### CPU

```bash
cd "$HOME/localrag"
sudo systemctl start docker
./scripts/compose.sh up -d
./scripts/compose.sh ps
```

### GPU

```bash
cd "$HOME/localrag"
sudo systemctl start docker

./scripts/compose.sh \
  -f compose.yaml \
  -f compose.gpu.yaml \
  up -d

./scripts/compose.sh \
  -f compose.yaml \
  -f compose.gpu.yaml \
  ps
```

The services also have `restart: unless-stopped`, so previously running containers may already restart when Docker starts.

## Apply Python, HTML, or CSS changes

Application code is copied into the image, not mounted live. **Restarting alone does not apply code changes.**

### CPU

```bash
cd "$HOME/localrag"

./scripts/compose.sh build web worker
./scripts/compose.sh up -d --force-recreate web worker
```

### GPU

```bash
cd "$HOME/localrag"

./scripts/compose.sh \
  -f compose.yaml \
  -f compose.gpu.yaml \
  build web worker

./scripts/compose.sh \
  -f compose.yaml \
  -f compose.gpu.yaml \
  up -d --force-recreate web worker
```

Database model changes also require proper migrations; rebuilding does not update an existing database schema automatically.

## Stop the containers without deleting them

### CPU

```bash
./scripts/compose.sh stop
```

### GPU

```bash
./scripts/compose.sh \
  -f compose.yaml \
  -f compose.gpu.yaml \
  stop
```

## Remove containers while preserving persistent volumes

### CPU

```bash
./scripts/compose.sh down
```

### GPU

```bash
./scripts/compose.sh \
  -f compose.yaml \
  -f compose.gpu.yaml \
  down
```

You can recreate them with `up -d`.

**Do not add `-v` to `down` unless you intentionally want to delete the project’s named volumes**, including the database, downloaded models, and monitoring history.

## If startup fails

Start with:

```bash
cd "$HOME/localrag"

./scripts/compose.sh ps -a
./scripts/compose.sh logs --tail=200 db web worker
```

For a database connection error:

```bash
./scripts/compose.sh exec -T db \
  pg_isready -h 127.0.0.1 -U localrag -d localrag
```

If a container is not running, `exec` cannot enter it—use its logs to identify the failure first. Avoid deleting volumes as a troubleshooting shortcut.
