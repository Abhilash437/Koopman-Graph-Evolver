# Contributing to Koopman Graph Evolver

First off, thank you for considering contributing to Koopman Graph Evolver! It's people like you that make open source such a great community.

## Where do I go from here?

If you've noticed a bug or have a feature request, make sure to check our [Issues](../../issues) first to see if someone else has already created a ticket. If not, go ahead and make one!

## How to Contribute

### 1. Fork the Repository
Fork the project to your own GitHub account and clone it to your local machine.

### 2. Set up your Environment
We recommend using a virtual environment (like `venv` or `conda`) or the provided Docker container to manage dependencies.
```bash
pip install -r requirements.txt
```

### 3. Create a Branch
Create a new branch for your feature or bugfix:
```bash
git checkout -b feature/your-feature-name
```

### 4. Make your Changes
Make sure your code adheres to the existing style and includes adequate docstrings. If you are adding a new model or ablation suite, please update the CLI (`koopman_evolver/cli.py`) and GUI (`app.py`) accordingly.

### 5. Commit and Push
Commit your changes with a descriptive commit message. We follow the Conventional Commits specification (e.g., `feat:`, `fix:`, `docs:`).
```bash
git commit -m "feat: Add new molecular dynamics baseline"
git push origin feature/your-feature-name
```

### 6. Submit a Pull Request
Open a Pull Request against the `main` branch. Fill out the PR template completely to help reviewers understand your changes.

## Code of Conduct
By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). We expect all contributors to adhere to it.
