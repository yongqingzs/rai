# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Project Overview

RAI (RobotecAI) is a Python monorepo providing an Embodied AI agent framework that integrates LLMs (via LangChain/LangGraph) with ROS 2 for robotic multi-agent systems. It supports Ubuntu 22.04/24.04, Python 3.10/3.12, and ROS 2 Humble/Jazzy.

## Monorepo Layout

The repository uses `uv` for Python dependency management with a workspace-style setup. Each sub-package in `src/` has its own `pyproject.toml`. Local packages are declared as editable sources in the root `pyproject.toml`.

| Package | Purpose |
|---------|---------|
| `src/rai_core` | Core `rai` module: agents (ReAct, state-based, ROS2), LangChain integration, tools, aggregators, communication (HRI), LLM initialization |
| `src/rai_whoami` | Robot embodiment info extraction from docs/URDFs |
| `src/rai_s2s` | Speech-to-speech (ASR/TTS) pipeline |
| `src/rai_sim` | Simulator connection |
| `src/rai_bench` | Benchmarking suite |
| `src/rai_extensions/rai_perception` | Object detection with open-set ML models |
| `src/rai_extensions/rai_nomad` | NoMaD navigation integration |
| `src/rai_semap` | Semantic map module |
| `src/rai_finetune` | LLM fine-tuning with Unsloth (alpha, Python 3.10 only) |
| `src/rai_bringup` | ROS 2 launch files for deployment |
| `src/rai_interfaces` | ROS 2 msg/srv/action definitions (CMake-based, cloned via `vcs import`) |
| `examples/` | Demo scripts (manipulation, agriculture, ROSbot XL) |

The core `rai` module (from `src/rai_core`) exports `AgentRunner`, `ReActAgent`, `BaseAgent`, `BaseStateBasedAgent`, `wait_for_shutdown`, `get_llm_model`, `get_embeddings_model`, `get_tracing_callbacks`, and `timeout`.

## Environment Setup

```bash
# Install dependencies (all groups needed for full test collection)
uv sync --all-groups

# Import ROS 2 dependencies (includes rai_interfaces)
vcs import src < ros_deps.repos
rosdep install --from-paths src --ignore-src -r -y

# Build ROS 2 workspace
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install

# Source the environment (activates .venv + colcon overlay + PYTHONPATH)
source ./setup_shell.sh
```

`setup_shell.sh` activates `.venv`, sources `install/setup.bash`, and sets `PYTHONPATH` to include `src/rai_core`, `src/rai_sim`, `src/rai_s2s`, and `src/rai_bench`. **Always source this before running tests or Python scripts.** ROS 2 must be sourced *before* calling setup_shell.sh.

## Common Commands

```bash
# Lint and format (ruff + prettier + pre-commit-hooks via pre-commit)
pre-commit run --all-files

# Run individual pre-commit hooks
pre-commit run ruff --all-files
pre-commit run ruff-format --all-files

# Run full test suite (local, non-billable, non-manual)
pytest tests/ --timeout=60

# Run a single test file
pytest tests/agents/test_some_file.py --timeout=60

# Run a specific test function
pytest tests/agents/test_some_file.py::test_function_name --timeout=60

# Run tests with coverage
pytest tests/ --cov=./src/rai_core --cov-report=xml --timeout=60

# Skip sound device tests (require audio hardware)
pytest tests/ --ignore=tests/communication/sounds_device --timeout=60
```

Pytest markers: `billable` (costs money), `ci_only` (CI-only), `manual` (needs running demo app). Default `addopts` excludes all three and adds `--ignore=src`. Custom CLI option `--strategy` (default `centroid`) is available for gripping point detection tests.

## Architecture Highlights

- **Agent hierarchy**: `BaseAgent` (abstract) → `LangChainAgent` → `ReActAgent` / `StateBasedAgent` / `ROS2StateBasedAgent`. Agents are orchestrated by `AgentRunner` with signal-safe shutdown via `wait_for_shutdown`.
- **LangChain 1.x core**: Under `rai/agents/langchain/core/` there are additional agent types: `ConversationalAgent`, `StructuredOutputAgent`, `PlanAgent`, `MegamindAgent`, and `ToolRunner`.
- **LLM abstraction**: `get_llm_model()` / `get_embeddings_model()` read `config.toml` to select vendor (OpenAI, AWS Bedrock, Ollama, Google) and return LangChain model instances. Tracing via Langfuse/LangSmith is configured in the `[tracing]` sections.
- **ROS 2 bridge**: `ROS2StateBasedAgent` extends state-based agents with ROS 2 node lifecycle. The `rai/communication/ros2/` module provides ROS 2 API wrappers for topics, services, and actions via connectors. `rai_interfaces` provides custom msg/srv/action types built via CMake/colcon.
- **Configuration**: `config.toml` at repo root is the active config. It defines vendor selections, model names, ASR/TTS settings, and tracing. A Streamlit configurator GUI exists at `rai/frontend/configurator.py` but all config can be edited directly in the TOML file.
- **Aggregators**: `rai/aggregators/` provides data aggregation for agents, with both base and ROS 2 variants for collecting sensor data, map info, and navigation state.

## Key Gotchas

- The `rai` module is installed from `src/rai_core` with `module-name = "rai"` (not `rai_core`). Import as `from rai.agents import ...`.
- Many test files import `rclpy` and ROS 2 msgs at module level — ROS 2 **must** be sourced for test collection, not just execution.
- Tests under `tests/communication/sounds_device/` require audio devices. Skip with `--ignore=tests/communication/sounds_device` on headless systems.
- `config.toml` is tracked in git and modified during dev — the `test_config_toml` conftest fixture creates temp configs for tests.
- Circular imports between `rai.perception` and `rai_core` have been a historical issue (see commit a791cfd).
- `setup_shell.sh` does NOT add `rai_whoami`, `rai_semap`, or `rai_perception` to `PYTHONPATH` — these are installed as editable packages via uv and accessed through their own module names.
- CI runs on a matrix of Humble and Jazzy, with coverage uploaded only for Jazzy.

## Commit Conventions

Conventional Commits: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`. Example: `feat(agents): add custom tool schema support`.
