# Repository Guidelines

## Project Structure & Module Organization
- `raps/`: Core simulator, schedulers, dataloaders, network models, and workloads.
- `main.py`: CLI entry point (installed as `raps`).
- `config/`: System and partition YAML configs (e.g., `config/frontier.yaml`, `config/setonix/part-gpu.yaml`).
- `experiments/`: Example experiment YAMLs for repeatable runs.
- `tests/`: Pytest suite with unit and system tests (`tests/unit/`, `tests/systems/`).
- `scripts/`: Helper scripts for plotting, data extraction, and experiments.
- `models/`: FMU models and cooling assets (see `make fetch-example-fmus`).

## Build, Test, and Development Commands
- `source /opt/venvs/exadigit/bin/activate`: Activate the recommended virtualenv if available.
- `pip install -e .`: Install in editable mode (Python 3.12+).
- `raps run -h`: Show CLI help and options.
- `raps run`: Run the default synthetic simulation.
- `make test` or `pytest -n 8`: Run tests with xdist parallelism.
- `make docker_build` / `make docker_run`: Build and run the Docker image.
- `make fetch-example-fmus`: Download example cooling FMUs into `models/POWER9CSM`.

## Coding Style & Naming Conventions
- Python uses 4-space indentation and PEP 8-style naming.
- Modules/functions use `snake_case`; classes use `CapWords`.
- Keep CLI flags consistent with existing commands (see `main.py` and `raps/`).
- If you add scripts, place them in `scripts/` with descriptive names (e.g., `run_*`, `plot_*`).

## Testing Guidelines
- Framework: `pytest` with markers defined in `pytest.ini` (e.g., `unit`, `system`, `network`).
- Test files follow `test_*.py` and live under `tests/`.
- Data-backed tests require `RAPS_DATA_DIR` (e.g., `RAPS_DATA_DIR=/opt/data pytest -n auto -x`).
- Use marker filtering for scope, e.g., `pytest -m network` or `pytest -k multi_part_sim`.

## Commit & Pull Request Guidelines
- Recent history favors short, imperative commit messages (e.g., "Add ...", "Fix ...").
- Keep commits focused; describe data assumptions or configs in the body when relevant.
- PRs should include a brief summary, testing commands run, and any data dependencies.

## Configuration & Data Notes
- System and partition configs live in `config/` and are referenced by `--system` or `-x`.
- Telemetry replays often require external datasets in `/opt/data` or a custom path.
- Simulation outputs are commonly written to `raps-output-*`; avoid committing generated data.
