from huggingface_hub import snapshot_download
from affine.setup import logger

@cli.command("pull")
@click.argument("uid", type=int)
@click.option(
    "--model_path",
    "-p",
    default="./model_path",
    required=True,
    type=click.Path(),
    help="Local directory to save the model",
)
@click.option(
    "--hf-token", default=None, help="Hugging Face API token (env HF_TOKEN if unset)"
)
def pull(uid: int, model_path: str, hf_token: str):
    hf_token = hf_token or get_conf("HF_TOKEN")

    miner_map = asyncio.run(miners(uids=uid))
    miner = miner_map.get(uid)

    if miner is None:
        click.echo(f"No miner found for UID {uid}", err=True)
        sys.exit(1)
    repo_name = miner.model
    logger.info("Pulling model %s for UID %d into %s", repo_name, uid, model_path)

    try:
        snapshot_download(
            repo_id=repo_name,
            repo_type="model",
            local_dir=model_path,
            token=hf_token,
            resume_download=True,
            revision=miner.revision,
        )
        click.echo(f"Model {repo_name} pulled to {model_path}")
    except Exception as e:
        logger.error("Failed to download %s: %s", repo_name, e)
        click.echo(f"Error pulling model: {e}", err=True)
        sys.exit(1)


@cli.command("chutes_push")
@click.option("--repo", required=True, help="Existing HF repo id (e.g. <user>/<repo>)")
@click.option("--revision", required=True, help="HF commit SHA to deploy")
@click.option(
    "--chutes-api-key",
    default=None,
    help="Chutes API key (env CHUTES_API_KEY if unset)",
)
def push(repo: str, revision: str, chutes_api_key: str, chute_user: str):
    """Deploy an existing HF repo+revision to Chutes and print the chute info."""
    chutes_api_key = chutes_api_key or get_conf("CHUTES_API_KEY")
    chute_user = chute_user or get_conf("CHUTE_USER")

    async def deploy_to_chutes():
        logger.debug("Building Chute config for repo=%s revision=%s", repo, revision)
        chutes_config = textwrap.dedent(
            f"""
import os
from chutes.chute import NodeSelector
from chutes.chute.template.sglang import build_sglang_chute
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

chute = build_sglang_chute(
    username="{chute_user}",
    readme="{repo}",
    model_name="{repo}",
    image="chutes/sglang:nightly-2025081600",
    concurrency=20,
    revision="{revision}",
    node_selector=NodeSelector(
        gpu_count=1,
        include=["a100", "h100"],
    ),
    max_instances=1,
    scale_threshold=0.5,
    shutdown_after_seconds=3600,
)
"""
        )
        tmp_file = Path("tmp_chute.py")
        tmp_file.write_text(chutes_config)
        logger.debug("Wrote Chute config to %s", tmp_file)

        cmd = ["chutes", "deploy", f"{tmp_file.stem}:chute", "--accept-fee"]
        env = {**os.environ, "CHUTES_API_KEY": chutes_api_key}
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.PIPE,
        )
        if proc.stdin:
            proc.stdin.write(b"y\n")
            await proc.stdin.drain()
            proc.stdin.close()
        stdout, _ = await proc.communicate()
        output = stdout.decode(errors="ignore")
        logger.trace(output)
        import re

        match = re.search(
            r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3})\s+\|\s+(\w+)", output
        )
        if match and match.group(2) == "ERROR":
            logger.debug("Chutes deploy failed with the above error log")
            raise RuntimeError("Chutes deploy failed")
        if proc.returncode != 0:
            logger.debug("Chutes deploy failed with code %d", proc.returncode)
            raise RuntimeError("Chutes deploy failed")
        tmp_file.unlink(missing_ok=True)
        logger.debug("Chute deployment successful")

    asyncio.run(deploy_to_chutes())

    chute_id = asyncio.run(get_latest_chute_id(repo, api_key=chutes_api_key))
    chute_info = asyncio.run(get_chute(chute_id)) if chute_id else None
    payload = {
        "success": bool(chute_id),
        "chute_id": chute_id,
        "chute": chute_info,
        "repo": repo,
        "revision": revision,
    }
    click.echo(json.dumps(payload))


@cli.command("commit")
@click.option("--repo", required=True, help="HF repo id (e.g. <user>/<repo>)")
@click.option("--revision", required=True, help="HF commit SHA")
@click.option("--chute-id", required=True, help="Chutes deployment id")
@click.option("--coldkey", default=None, help="Name of the cold wallet to use.")
@click.option("--hotkey", default=None, help="Name of the hot wallet to use.")
def commit(repo: str, revision: str, chute_id: str, coldkey: str, hotkey: str):
    """Commit repo+revision+chute_id on-chain (separate from deployment)."""
    cold = coldkey or get_conf("BT_WALLET_COLD", "default")
    hot = hotkey or get_conf("BT_WALLET_HOT", "default")
    wallet = bt.wallet(name=cold, hotkey=hot)

    async def _commit():
        sub = await get_subtensor()
        data = json.dumps({"model": repo, "revision": revision, "chute_id": chute_id})
        while True:
            try:
                await sub.set_reveal_commitment(
                    wallet=wallet, netuid=NETUID, data=data, blocks_until_reveal=1
                )
                break
            except MetadataError as e:
                if "SpaceLimitExceeded" in str(e):
                    await sub.wait_for_block()
                else:
                    raise

    try:
        asyncio.run(_commit())
        click.echo(
            json.dumps(
                {
                    "success": True,
                    "repo": repo,
                    "revision": revision,
                    "chute_id": chute_id,
                }
            )
        )
    except Exception as e:
        logger.error("Commit failed: %s", e)
        click.echo(json.dumps({"success": False, "error": str(e)}))

