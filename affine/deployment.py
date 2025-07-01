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
from . import val_config

class DeploymentConfig:
    """Configuration container for deployment parameters"""
    def __init__(self, **kwargs):
        self.chutes_api_key = kwargs.get('chutes_api_key')
        self.hf_user = kwargs.get('hf_user')
        self.hf_token = kwargs.get('hf_token')
        self.chute_user = kwargs.get('chute_user')
        self.wallet_cold = kwargs.get('wallet_cold')
        self.wallet_hot = kwargs.get('wallet_hot')


async def deploy_model(
    local_path: str,
    config: DeploymentConfig,
    existing_repo: Optional[str] = None,
    blocks_until_reveal: int = 1
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
    await deploy_to_chutes(repo_id, config.chute_user, config.chutes_api_key)
    await test_deployment(repo_id, config.chutes_api_key)
    
    return repo_id


def _generate_repo_name(hf_user: str) -> str:
    """Generate a unique repository name"""
    random_number = random.randint(1000000, 9999999)
    return f"{hf_user}/Affine-{random_number}"


async def push_to_huggingface(local_path: str, repo_id: str, hf_token: str) -> None:
    """Push model to HuggingFace Hub"""
    print(f"Pushing model from {local_path} to {repo_id}")
    hf_api = HfApi(token=hf_token)
    
    # Create or verify repository
    try:
        hf_api.repo_info(repo_id=repo_id, repo_type="model")
        print(f"✓ Repository exists: {repo_id}")
    except Exception:
        print(f"✓ Creating new repository: {repo_id}")
        hf_api.create_repo(repo_id=repo_id, private=False, repo_type="model", exist_ok=True)
    
    # Collect files to upload
    files = []
    for root, _, filenames in os.walk(local_path):
        if '.cache' in root or any(part.startswith('.') for part in root.split(os.sep)):
            continue
        for filename in filenames:
            if not filename.startswith('.') and not filename.endswith('.lock'):
                files.append(os.path.join(root, filename))
    
    # Upload files
    print(f"Uploading {len(files)} files...")
    for file_path in files:
        relative_path = os.path.relpath(file_path, local_path)
        hf_api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=relative_path,
            repo_id=repo_id,
            repo_type="model"
        )
        print(f"  ✓ {relative_path}")
    
    print(f"Successfully pushed to {repo_id}")


async def push_to_chain(repo_id: str, blocks_until_reveal: int, wallet_cold: str, wallet_hot: str) -> None:
    """Push repository to Bittensor chain"""
    print(f"⛓️  Pushing {repo_id} to Bittensor chain...")
    
    sub = bt.subtensor()
    wallet = bt.wallet(name=wallet_cold, hotkey=wallet_hot)
    
    sub.set_reveal_commitment(
        wallet=wallet,
        netuid=val_config.BITTENSOR_NETUID,
        data=repo_id,
        blocks_until_reveal=blocks_until_reveal
    )
    
    print(f"Committed to chain ({blocks_until_reveal} blocks until reveal)")


async def deploy_to_chutes(repo_id: str, chute_user: str, chutes_api_key: str) -> None:
    """Deploy model to Chutes platform"""
    print(f"Deploying {repo_id} to Chutes...")
    
    temp_chute_name = "temp_chute"
    temp_chute_file = f"{temp_chute_name}.py"
    
    chute_config = _generate_chute_config(repo_id, chute_user)
    
    try:
        with open(temp_chute_file, 'w') as f:
            f.write(chute_config)
        
        deploy_cmd = f"chutes deploy {temp_chute_name}:chute --public"
        result = os.system(deploy_cmd)
        
        if result != 0:
            raise RuntimeError(f"Chutes deploy failed with status {result}")
        
        print(" Successfully deployed to Chutes")
        
    finally:
        if os.path.exists(temp_chute_file):
            os.remove(temp_chute_file)


def _generate_chute_config(repo_id: str, chute_user: str) -> str:
    """Generate Chutes configuration file content"""
    return f'''import os
from chutes.chute import NodeSelector
from chutes.chute.template.sglang import build_sglang_chute

os.environ["NO_PROXY"] = "localhost,127.0.0.1"

chute = build_sglang_chute(
    username="{chute_user}",
    readme="{repo_id}",
    model_name="{repo_id}",
    image="chutes/sglang:0.4.6.post5b",
    concurrency=20,
    node_selector=NodeSelector(
        gpu_count=8,
        min_vram_gb_per_gpu=24,
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
    print("Testing deployment...")
    
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
            ) as r:
                await r.text()
                print("✅ Deployment test successful!")
        except Exception as e:
            print(f"Note: Expected timeout during warm-up: {str(e)}")
            print("Deployment test completed!")