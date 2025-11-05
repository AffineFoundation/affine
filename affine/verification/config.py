"""Configuration loader for verification service."""

import os
from pathlib import Path
from dotenv import load_dotenv


def load_verification_env():
    """Load verification environment variables from .env file."""
    # Try to load from verification directory first
    verification_env = Path(__file__).parent / ".env"

    if verification_env.exists():
        load_dotenv(verification_env, override=False)
        return True

    # Fall back to main .env if exists
    main_env = Path(__file__).parent.parent.parent / ".env"
    if main_env.exists():
        load_dotenv(main_env, override=False)
        return True

    return False


# Auto-load on import
load_verification_env()


def get_verification_config():
    """Get verification configuration as a dictionary."""
    return {
        # GPU Configuration
        "gpu_mode": os.getenv("GPU_MODE", "remote"),
        "gpu_host": os.getenv("GPU_HOST", ""),
        "gpu_ssh_key": os.getenv("GPU_SSH_KEY", ""),
        "gpu_work_dir": os.getenv("GPU_WORK_DIR", "/opt/affine/models"),

        # SGLang Configuration
        "sglang_image": os.getenv("SGLANG_DOCKER_IMAGE", "lmsysorg/sglang:latest"),
        "port_range_start": int(os.getenv("GPU_PORT_RANGE_START", "30000")),
        "port_range_end": int(os.getenv("GPU_PORT_RANGE_END", "31000")),

        # Monitor Configuration
        "monitor_interval": int(os.getenv("VERIFICATION_MONITOR_INTERVAL", "300")),

        # Verification Configuration
        "sample_size": int(os.getenv("VERIFICATION_SAMPLE_SIZE", "10")),
        "similarity_threshold": float(os.getenv("VERIFICATION_SIMILARITY_THRESHOLD", "0.85")),
        "similarity_metric": os.getenv("VERIFICATION_SIMILARITY_METRIC", "token_overlap"),
        "max_retries": int(os.getenv("VERIFICATION_MAX_RETRIES", "3")),
        "sample_tail": int(os.getenv("VERIFICATION_SAMPLE_TAIL", "10000")),

        # Required from main config
        "hf_token": os.getenv("HF_TOKEN", ""),
        "chutes_api_key": os.getenv("CHUTES_API_KEY", ""),
        "r2_folder": os.getenv("R2_FOLDER", "affine"),
        "r2_bucket_id": os.getenv("R2_BUCKET_ID", ""),
        "r2_access_key": os.getenv("R2_WRITE_ACCESS_KEY_ID", ""),
        "r2_secret_key": os.getenv("R2_WRITE_SECRET_ACCESS_KEY", ""),
    }


def print_verification_config():
    """Print verification configuration (for debugging)."""
    config = get_verification_config()

    print("Verification Service Configuration:")
    print("=" * 50)

    print("\nGPU Configuration:")
    print(f"  Mode:           {config['gpu_mode']}")
    if config['gpu_mode'] == 'remote':
        print(f"  GPU Host:       {config['gpu_host']}")
        print(f"  SSH Key:        {config['gpu_ssh_key'] or '(default)'}")
    print(f"  Work Dir:       {config['gpu_work_dir']}")

    print("\nSGLang Configuration:")
    print(f"  Docker Image:   {config['sglang_image']}")
    print(f"  Port Range:     {config['port_range_start']}-{config['port_range_end']}")

    print("\nMonitor Configuration:")
    print(f"  Interval:       {config['monitor_interval']}s")

    print("\nVerification Configuration:")
    print(f"  Sample Size:    {config['sample_size']}")
    print(f"  Threshold:      {config['similarity_threshold']}")
    print(f"  Metric:         {config['similarity_metric']}")
    print(f"  Max Retries:    {config['max_retries']}")
    print(f"  Sample Tail:    {config['sample_tail']}")

    print("\nExternal Services:")
    print(f"  HF Token:       {'✓ Set' if config['hf_token'] else '✗ Not set'}")
    print(f"  Chutes API:     {'✓ Set' if config['chutes_api_key'] else '✗ Not set'}")
    print(f"  R2 Bucket:      {config['r2_folder']}")
    print(f"  R2 Credentials: {'✓ Set' if config['r2_access_key'] and config['r2_secret_key'] else '✗ Not set'}")

    print("=" * 50)
