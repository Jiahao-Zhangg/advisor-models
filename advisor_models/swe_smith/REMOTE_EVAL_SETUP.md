# Remote Evaluation Setup for SWE-Smith

This guide explains how to set up remote evaluation for SWE-Smith when running on a cluster that doesn't support Docker-in-Docker.

## Architecture

```
┌─────────────────┐         HTTP          ┌──────────────────┐
│  Training       │ ───────────────────> │   External VM    │
│  Cluster        │   Evaluation Request  │  (Docker Server) │
│  (Training/Eval)│ <─────────────────── │                  │
└─────────────────┘    Evaluation Result  └──────────────────┘
```

## Setup Instructions

### 1. Set Up the External VM

You need a VM with:
- Docker installed and running
- Python 3.10+
- Network access from training cluster
- Open port (default: 8080)

#### Install Dependencies on VM

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/advisor-models.git
cd advisor-models

# Install dependencies
pip install uv
uv sync

# Install Flask for the server
uv add flask requests

# Install SWE-Smith harness (requires Docker)
git clone https://github.com/SWE-bench/SWE-smith /tmp/swe-smith
cd /tmp/swe-smith

# Install build dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y pkg-config libssl-dev

# Install Rust (required for some dependencies)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env

# Install SWE-Smith into advisor-models venv
uv pip install -e . --python ~/advisor-models/.venv/bin/python
cd ~/advisor-models/
```

#### Start the Evaluation Server

```bash
cd advisor-models
source .venv/bin/activate

# Make sure Docker is running
docker ps

# Start the server (runs on port 8080 by default)
uv run python -m advisor_models.swe_smith.eval_server --host 0.0.0.0 --port 8080 --workers 64

# Or run in background with nohup
nohup uv run python -m advisor_models.swe_smith.eval_server --host 0.0.0.0 --port 8080 --workers 64 > eval_server.log 2>&1 &
```