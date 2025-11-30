# Deployment Guide

This project publishes three components via GitHub releases: `migrate`, `server`, and `client` binaries, plus datasets.

## Quick Start

All components can be downloaded and run using simple shell scripts from GitHub.

### Server VM - Deploy Server

```bash
bash <(wget -qO- https://raw.githubusercontent.com/jakob-lilliemarck/fx-fitness-search/main/scripts/download-server.sh) v1.0.0 /opt/app
```

Or step by step:
```bash
wget -O download-server.sh https://raw.githubusercontent.com/jakob-lilliemarck/fx-fitness-search/main/scripts/download-server.sh
chmod +x download-server.sh
./download-server.sh v1.0.0 /opt/app
```

This downloads:
- `server` binary
- `data/` directory with all datasets

### Client Machine - Deploy Client

```bash
bash <(wget -qO- https://raw.githubusercontent.com/jakob-lilliemarck/fx-fitness-search/main/scripts/download-client.sh) v1.0.0
```

Or step by step:
```bash
wget -O download-client.sh https://raw.githubusercontent.com/jakob-lilliemarck/fx-fitness-search/main/scripts/download-client.sh
chmod +x download-client.sh
./download-client.sh v1.0.0
```

This downloads the `client` binary to the current directory.

## Version Management

Replace `v1.0.0` with any released version tag. To use the latest release, omit the version:

```bash
./download-server.sh . v1.0.0   # Specific version
./download-client.sh v1.0.0     # Specific version
```

## Datasets

Datasets are published separately and don't change with each release. The `download-server.sh` script automatically fetches and extracts them.

To manually download datasets only:
```bash
wget https://github.com/jakob-lilliemarck/fx-fitness-search/releases/download/data-beijing-v1/beijing-air-quality.tar.gz
tar -xzf beijing-air-quality.tar.gz
```

## Release Workflow

**Binaries** (server, client, migrate):
1. Commit and push your code
2. Go to GitHub → Actions → "Release Binaries"
3. Click "Run workflow"
4. Enter version tag (e.g., `v1.0.0`)
5. Binaries are published to that release tag

**Datasets** (one-time):
1. Go to GitHub → Actions → "Publish Datasets"
2. Click "Run workflow"
3. Datasets are published to their configured tags

## Database Setup

Database migrations are handled automatically by Docker Compose. See `db/README.md` for local database setup and deployment instructions.
