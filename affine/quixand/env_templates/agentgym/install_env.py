#!/usr/bin/env python3

import subprocess
import sys
import os
from pathlib import Path

def install_miniconda(env_yaml_path):
    """Install Miniconda if not already present."""
    conda_path = Path("/opt/miniconda")
    if conda_path.exists():
        print("Miniconda already installed")
        return str(conda_path / "bin" / "conda")
    
    print("Installing Miniconda...")
    subprocess.run([
        "wget", "-q",
        "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh",
        "-O", "/tmp/miniconda.sh"
    ], check=True)
    
    subprocess.run([
        "bash", "/tmp/miniconda.sh", "-b", "-p", str(conda_path)
    ], check=True)
    
    subprocess.run(["rm", "/tmp/miniconda.sh"], check=True)
    
    conda_bin = str(conda_path / "bin" / "conda")
    
    print("Accepting conda terms of service...")
    subprocess.run([
        conda_bin, "tos", "accept", "--override-channels",
        "--channel", "https://repo.anaconda.com/pkgs/main"
    ], check=False)
    
    subprocess.run([
        conda_bin, "tos", "accept", "--override-channels",
        "--channel", "https://repo.anaconda.com/pkgs/r"
    ], check=False)
    
    print(f"Setting up conda environment from {env_yaml_path}")
    
    subprocess.run([
        "conda", "env", "create", "--yes", "-f", str(env_yaml_path), "-n", f"agentenv"
    ], check=False)


def install_environment():
    """Install AgentGym environment based on ENV_NAME."""
    env_name = os.environ.get('ENV_NAME', '')
    env = os.environ.copy()
    
    if not env_name:
        print("No ENV_NAME specified, skipping environment installation")
        return

    print(f"Installing AgentGym environment: {env_name}")

    if env_name == "alfworld":
        env["CFLAGS"] = "-O3 -fPIC -Wno-incompatible-pointer-types"
        env["CPPFLAGS"] = "-Wno-incompatible-pointer-types"

    env_path = Path(f"/app/AgentGym/agentenv-{env_name}")

    env_yaml_path = env_path / "environment.yml"
    if env_yaml_path.exists():
        print(f"Found {env_yaml_path.name} for {env_name}, setting up conda environment...")
        install_miniconda(env_yaml_path)

    setup_script = env_path / "setup.sh"
    if setup_script.exists():
        print(f"Found setup.sh for {env_name}, executing...")
        try:
            setup_script.chmod(0o755)
            env["PIP_NO_INPUT"]="1"
            
            subprocess.run(
                ["bash", str(setup_script)],
                cwd=str(env_path),
                env=env,
                text=True
            )

        except Exception as e:
            print(f"Warning: Failed to execute setup.sh: {e}")
    else:
        if env_path.exists():
            subprocess.run([
                "pip", "install", "-e", str(env_path), "--no-build-isolation"
            ], text=True)

    if env_name == "webshop":
        subprocess.run([
            "pip", "install", "--force-reinstall", "typing-extensions==4.5.0"
        ], check=False)
    print(f"Environment {env_name} installation completed")

if __name__ == "__main__":
    install_environment()