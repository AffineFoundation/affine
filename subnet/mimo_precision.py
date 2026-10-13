"""Prospective, signed execution precision and worker-observed provenance.

This is a compatibility guard, not a tolerance adjustment or evidence that an
external miner ran a particular device. Historical runtime-v1 does not use it.
"""
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import re
import json
import os
import subprocess


class PrecisionMismatch(RuntimeError):
    """Execution-profile incompatibility, not an invalid-sample accusation."""


@dataclass(frozen=True)
class PrecisionProfile:
    version: str = "mimo-qualified-precision-v1"
    matmul_tf32: bool = False
    float32_precision: str = "highest"
    cudnn_tf32: bool = True
    bf16_reduced_precision_reduction: bool = True
    fp16_reduced_precision_reduction: bool = True
    deterministic_algorithms: bool = False


def profile_dict():
    return asdict(PrecisionProfile())


def validate_profile(value):
    expected = profile_dict()
    if (type(value) is not dict or set(value) != set(expected)
            or any(type(value[k]) is not type(v) or value[k] != v for k, v in expected.items())):
        raise ValueError("exact qualified MiMo precision profile required")
    return PrecisionProfile(**value)


def profile_digest(value):
    validate_profile(value)
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def observed(torch):
    return PrecisionProfile(
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        float32_precision=torch.get_float32_matmul_precision(),
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
    )


def require_no_autocast(torch):
    # The qualified package supports explicit device arguments. An unsupported
    # API must fail qualification rather than silently omitting a check.
    if torch.is_autocast_enabled("cuda") or torch.is_autocast_enabled("cpu"):
        raise PrecisionMismatch("ambient autocast is outside the precision profile")


def require_launch_profile():
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise PrecisionMismatch("CUBLAS workspace must be pinned before model/cache creation")


def guard(torch, expected=PrecisionProfile()):
    require_launch_profile()
    require_no_autocast(torch)
    if expected != PrecisionProfile():
        raise PrecisionMismatch("unqualified prospective precision profile")
    actual = observed(torch)
    if actual != expected:
        raise PrecisionMismatch("execution precision flags differ from signed profile")


def configure_before_model_load(torch, expected=PrecisionProfile()):
    """Call once in a dedicated process, before constructing model/cache state.

    This intentionally sets the signed execution policy. Forward calls only
    check; they never silently reset flags after state has been created.
    """
    require_launch_profile()
    require_no_autocast(torch)
    if expected != PrecisionProfile():
        raise PrecisionMismatch("unqualified prospective precision profile")
    torch.set_float32_matmul_precision(expected.float32_precision)
    torch.backends.cuda.matmul.allow_tf32 = expected.matmul_tf32
    torch.backends.cudnn.allow_tf32 = expected.cudnn_tf32
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = expected.bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = expected.fp16_reduced_precision_reduction
    torch.use_deterministic_algorithms(expected.deterministic_algorithms)
    guard(torch, expected)


def guarded_forward(torch, function, *args, expected=PrecisionProfile(), **kwargs):
    guard(torch, expected)
    result = function(*args, **kwargs)
    guard(torch, expected)
    return result


def provenance(torch, profile, *, device="cuda"):
    """Observe this trusted worker process; never copy submitted miner claims."""
    expected = validate_profile(profile)
    guard(torch, expected)
    hardware = None
    if str(device).startswith("cuda"):
        props = torch.cuda.get_device_properties(device)
        uuid = str(getattr(props, "uuid", "unavailable"))
        driver = None
        query_uuid = "GPU-" + uuid if re.fullmatch(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", uuid) else uuid
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader",
                                     "--id=" + query_uuid], capture_output=True, text=True, timeout=5, check=True)
            lines = result.stdout.strip().splitlines()
            if len(lines) == 1 and 0 < len(lines[0]) <= 128:
                driver = lines[0]
        except (OSError, subprocess.SubprocessError):
            pass  # Missing observation is explicit, never a guessed driver.
        hardware = dict(name=str(props.name), capability=list(torch.cuda.get_device_capability(device)),
                        total_memory=int(props.total_memory), uuid=uuid, driver=driver)
    device = "cuda" if str(device).startswith("cuda") else str(device)
    return dict(version="mimo-worker-runtime-provenance-v1", scope="worker-process-observation",
        precision_profile_sha256=profile_digest(profile), observed_precision=asdict(observed(torch)),
        autocast=dict(cpu=False, cuda=False), device=str(device), hardware=hardware,
        build=dict(torch=str(torch.__version__), torch_distribution=importlib.metadata.version("torch"), cuda=torch.version.cuda,
                   cudnn=torch.backends.cudnn.version()),
        launch=dict(cpu_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                    cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                    deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled()))


def validate_provenance(value, profile, *, device="cuda"):
    """Validate report structure/binding after its worker signature is checked."""
    expected_keys = {"version", "scope", "precision_profile_sha256", "observed_precision", "autocast",
                     "device", "hardware", "build", "launch"}
    if (type(value) is not dict or set(value) != expected_keys
            or value["version"] != "mimo-worker-runtime-provenance-v1"
            or value["scope"] != "worker-process-observation"
            or value["precision_profile_sha256"] != profile_digest(profile)
            or value["device"] != device or type(value["autocast"]) is not dict
            or set(value["autocast"]) != {"cpu", "cuda"}
            or any(value["autocast"][key] is not False for key in ("cpu", "cuda"))):
        raise ValueError("worker execution provenance binding")
    validate_profile(value["observed_precision"])
    h = value["hardware"]
    if device == "cuda":
        if (type(h) is not dict or set(h) != {"name", "capability", "total_memory", "uuid", "driver"}
                or any(type(h[k]) is not str or not 1 <= len(h[k]) <= 256 for k in ("name", "uuid"))
                or type(h["capability"]) is not list or len(h["capability"]) != 2
                or any(type(x) is not int or not 0 <= x <= 100 for x in h["capability"])
                or type(h["total_memory"]) is not int or not 0 < h["total_memory"] < 2**60
                or h["driver"] is not None and (type(h["driver"]) is not str or not 1 <= len(h["driver"]) <= 128)):
            raise ValueError("bounded worker hardware observation")
    elif device != "cpu" or h is not None:
        raise ValueError("worker provenance device")
    b, launch = value["build"], value["launch"]
    if (type(b) is not dict or set(b) != {"torch", "torch_distribution", "cuda", "cudnn"}
            or any(type(b[k]) is not str or not 1 <= len(b[k]) <= 128 for k in ("torch", "torch_distribution"))
            or b["cuda"] is not None and (type(b["cuda"]) is not str or not 1 <= len(b["cuda"]) <= 128)
            or b["cudnn"] is not None and (type(b["cudnn"]) is not int or not 0 <= b["cudnn"] < 10**9)
            or type(launch) is not dict or set(launch) != {"cpu_threads", "interop_threads", "cublas_workspace_config", "deterministic_warn_only"}
            or any(type(launch[k]) is not int or not 1 <= launch[k] <= 65536 for k in ("cpu_threads", "interop_threads"))
            or launch["cublas_workspace_config"] != ":4096:8"
            or type(launch["deterministic_warn_only"]) is not bool):
        raise ValueError("bounded worker build/launch observation")
    return value
