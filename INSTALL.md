# Installation Guide

This guide covers the different ways to install and run Affine:

- [Using `uv` (Recommended)](#install-using-uv)
- [Using `pip` and `venv` (Standard)](#install-using-pip-and-venv)
- [Using Docker](#install-using-docker)

## Install using uv

We rely on `uv` for fast, reliable package management.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/AffineFoundation/affine.git
cd affine
uv venv && source .venv/bin/activate && uv pip install -e .

# Verify
af --help
```

## Install using pip and venv

If you prefer the standard Python tooling (`pip` and `venv`), follow these steps.

**Prerequisites:** Python 3.11 or higher.

1. **Clone the repository:**

   ```bash
   git clone https://github.com/AffineFoundation/affine.git
   cd affine
   ```

2. **Create a virtual environment:**

   ```bash
   python3 -m venv .venv
   ```

3. **Activate the environment:**
   - On Linux/macOS:

     ```bash
     source .venv/bin/activate
     ```

   - On Windows:

     ```powershell
     .venv\Scripts\activate
     ```

4. **Install dependencies and the package:**
   This project uses `pyproject.toml` for dependency management.

   ```bash
   pip install --upgrade pip build
   pip install -e .
   ```

   *Note: The `-e .` flag installs the package in "editable" mode, meaning
   changes to the code will be reflected immediately without reinstalling.*

5. **Verify installation:**

   ```bash
   af --help
   ```

### Troubleshooting

- **Python Version Mismatch**: Ensure you are using Python 3.11+. Check with
  `python3 --version`.
- **Permission Errors**: If you see permission errors, ensure you are in the
  virtual environment. Avoid using `sudo pip install`.
- **Missing Dependencies**: On some systems, you might need build tools. On
  Ubuntu/Debian: `sudo apt install build-essential python3-dev`.

## Install using Docker

You can use Docker to run the validator or watchtower services without installing
Python dependencies locally.

### Using Docker Compose (Recommended)

1. **Configure environment:**
   Copy `.env.example` to `.env` and fill in your details (wallet info, etc.).

   ```bash
   cp .env.example .env
   ```

2. **Run with Compose:**

   ```bash
   docker-compose up -d
   ```

   This will start the validator and watchtower services.

### Building Manually

You can build the image directly using the provided `Dockerfile`.

```bash
docker build -t affine .
docker run --env-file .env affine
```
