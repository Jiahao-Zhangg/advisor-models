# Remote Agent Execution Setup for SWE-Smith

This guide explains how to set up remote agent execution for SWE-Smith when running on a cluster that doesn't support Docker.

## Architecture

```
┌─────────────────┐         HTTP          ┌──────────────────┐
│  Training       │ ───────────────────> │   External VM    │
│  Cluster        │   Agent Requests      │  (Agent Server)  │
│  (RL Training)  │ <─────────────────── │   + Docker       │
└─────────────────┘    Agent Responses    └──────────────────┘
                                                    │
                                          ┌─────────┴─────────┐
                                          │ Docker Containers │
                                          │ (Agent Sessions)  │
                                          └───────────────────┘
```

## Setup Instructions

### 1. Set Up the External VM

You need a VM with:
- Docker installed and running
- Python 3.10+
- Network access from training cluster
- Open port (default: 8081)
- Sufficient resources (recommend 32+ cores, 64GB+ RAM for parallel sessions)

#### Install Dependencies on VM

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/advisor-models.git
cd advisor-models

# Install dependencies
pip install uv
uv sync

source .venv/bin/activate

# Install Flask for the server
uv add flask requests

# Install mini-swe-agent
cd advisor_models/swe_smith/mini-swe-agent
uv pip install -e .
cd ../../../
```

#### Start the Agent Server

```bash
# Make sure Docker is running
docker ps

# Set your OpenAI API key
export OPENAI_API_KEY=your_api_key_here

# Start the server (runs on port 8081 by default)
uv run python -m advisor_models.swe_smith.agent_server --host 0.0.0.0 --port 8081 --log-dir skyrl-eval-logs

# Or run in background with nohup
nohup uv run python -m advisor_models.swe_smith.agent_server --host 0.0.0.0 --port 8081 > agent_server.log 2>&1 &

# To enable logging of agent trajectories, provide a path to the folder where the trajectories should be saved
uv run python -m advisor_models.swe_smith.agent_server --host 0.0.0.0 --port 8081 --log-dir /path/to/log_dir
```