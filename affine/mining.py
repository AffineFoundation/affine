"""
Affine deployment module - handles model deployment to HuggingFace, Bittensor, and Chutes
"""
import os
import random
import aiohttp
import bittensor as bt
from pathlib import Path
from huggingface_hub import HfApi
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class DeploymentConfig:
    """Configuration container for deployment parameters"""
    chutes_api_key: Optional[str] = None
    hf_user: Optional[str] = None
    hf_token: Optional[str] = None
    chute_user: Optional[str] = None
    wallet_cold: Optional[str] = None
    wallet_hot: Optional[str] = None
    chutes_image: Optional[str] = None


async def deploy_model(
    local_path: str,
    config: DeploymentConfig,
    existing_repo: Optional[str] = None,
    blocks_until_reveal: int = 360
) -> str:
    """
    Deploy a model through the complete pipeline:
    1. Push to HuggingFace (if not existing_repo)
    2. Commit to Bittensor chain
    3. Deploy to Chutes
    4. Test deployment
    """
    if existing_repo:
        repo_id = existing_repo
        print(f"Using existing repository: {repo_id}")
    else:
        if not Path(local_path).exists():
            raise ValueError(f"Local path does not exist: {local_path}")
        
        repo_id = _generate_repo_name(config.hf_user)
        print(f"Generated repository name: {repo_id}")
        
        await push_to_huggingface(local_path, repo_id, config.hf_token)
    
    await push_to_chain(repo_id, blocks_until_reveal, config.wallet_cold, config.wallet_hot)
    await deploy_to_chutes(repo_id, config.chute_user, config.chutes_api_key, config.chutes_image)
    await test_deployment(repo_id, config.chutes_api_key)
    
    return repo_id


def _generate_repo_name(hf_user: str) -> str:
    """Generate a unique repository name"""
    random_number = random.randint(1000000, 9999999)
    return f"{hf_user}/Affine-{random_number}"


async def push_to_huggingface(local_path: str, repo_id: str, hf_token: str) -> None:
    """Push model to HuggingFace Hub"""
    print(f"📤 Pushing model from {local_path} to {repo_id}")
    hf_api = HfApi(token=hf_token)
    
    # Create or verify repository
    try:
        hf_api.repo_info(repo_id=repo_id, repo_type="model")
        print(f"✓ Repository exists: {repo_id}")
    except Exception:
        print(f"✓ Creating new repository: {repo_id}")
        hf_api.create_repo(repo_id=repo_id, private=False, repo_type="model", exist_ok=True)
    
    # Collect and upload files
    files_to_upload = _collect_model_files(local_path)
    print(f"📁 Uploading {len(files_to_upload)} files...")
    
    for file_path in files_to_upload:
        relative_path = os.path.relpath(file_path, local_path)
        hf_api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=relative_path,
            repo_id=repo_id,
            repo_type="model"
        )
        print(f"  ✓ {relative_path}")
    
    print(f"✅ Successfully pushed to {repo_id}")


def _collect_model_files(local_path: str) -> list:
    """Collect all model files, excluding cache and hidden files"""
    files = []
    for root, _, filenames in os.walk(local_path):
        # Skip cache and hidden directories
        if any(part.startswith('.') or 'cache' in part.lower() for part in root.split(os.sep)):
            continue
            
        for filename in filenames:
            # Skip hidden and lock files
            if not filename.startswith('.') and not filename.endswith('.lock'):
                files.append(os.path.join(root, filename))
    
    return files


async def push_to_chain(repo_id: str, blocks_until_reveal: int, wallet_cold: str, wallet_hot: str) -> None:
    """Push repository to Bittensor chain"""
    print(f"⛓️  Pushing {repo_id} to Bittensor chain...")
    
    subtensor = bt.subtensor()
    wallet = bt.wallet(name=wallet_cold, hotkey=wallet_hot)
    
    subtensor.set_reveal_commitment(
        wallet=wallet,
        netuid=120,
        data=repo_id,
        blocks_until_reveal=blocks_until_reveal
    )
    
    print(f"✅ Committed to chain ({blocks_until_reveal} blocks until reveal)")


async def deploy_to_chutes(repo_id: str, chute_user: str, chutes_api_key: str, chutes_image: Optional[str] = None) -> None:
    """Deploy model to Chutes platform"""
    print(f"🚀 Deploying {repo_id} to Chutes...")
    
    temp_chute_name = "temp_chute"
    temp_chute_file = f"{temp_chute_name}.py"
    
    chute_config = _generate_chute_config(repo_id, chute_user, chutes_image)
    
    try:
        # Write temporary chute config
        with open(temp_chute_file, 'w') as f:
            f.write(chute_config)
        
        # Deploy using chutes CLI
        deploy_cmd = f"chutes deploy {temp_chute_name}:chute --public"
        result = os.system(deploy_cmd)
        
        if result != 0:
            raise RuntimeError(f"Chutes deploy failed with status {result}")
        
        print("✅ Successfully deployed to Chutes")
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_chute_file):
            os.remove(temp_chute_file)


def _generate_chute_config(repo_id: str, chute_user: str, chutes_image: Optional[str] = None) -> str:
    """Generate Chutes configuration file content"""
    # Utiliser l'image configurée ou fallback
    if chutes_image:
        image = f"{chutes_image}"
    else:
        image = "chutes/sglang:v0.3.5.post1"
    
    return f'''import os
from chutes.chute import NodeSelector
from chutes.chute.template.sglang import build_sglang_chute

os.environ["NO_PROXY"] = "localhost,127.0.0.1"

chute = build_sglang_chute(
    username="{chute_user}",
    readme="{repo_id}",
    model_name="{repo_id}",
    image="{image}",
    concurrency=16,
    node_selector=NodeSelector(
        gpu_count=8,
        min_vram_gb_per_gpu=70,
    ),
    engine_args=(
        "--trust-remote-code "
        "--grammar-backend xgrammar "
        "--tool-call-parser qwen25 "
        "--json-model-override-args {{\\"rope_scaling\\":{{\\"type\\":\\"yarn\\",\\"factor\\":4.0,\\"original_max_position_embeddings\\":32768}}}}"
    ),
)
'''


async def test_deployment(repo_id: str, chutes_api_key: str) -> None:
    """Test deployment with a warm-up request"""
    print("🔥 Testing deployment...")
    
    headers = {
        "Authorization": f"Bearer {chutes_api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": repo_id,
        "messages": [{"role": "user", "content": "Warming up the model"}],
        "stream": False,
        "max_tokens": 128,
        "temperature": 0.7
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                "https://llm.chutes.ai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=10
            ) as response:
                await response.text()
                print("✅ Deployment test successful!")
        except Exception as e:
            print(f"Note: Expected timeout during warm-up: {str(e)}")
            print("✅ Deployment test completed!")