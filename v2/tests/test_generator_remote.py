"""Remote Generator scorer transport, tested without a network or secrets."""

from __future__ import annotations

import io
import json
import gzip
import shlex
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from sn120_v2.errors import ConfigError, InfraError
from sn120_v2.generator.remote import ScorerRemote, SSHConfig, SSHRunner

EPOCH = "20261120T0000Z"
ROOT = f"/root/gen/jobs/{EPOCH}"


class FakeRemote:
    """The same small interface as SSHRunner, with in-memory pod state."""

    def __init__(self, config: SSHConfig):
        self.config = config
        self.destination = f"{config.user}@{config.host}"
        self.calls: list[tuple[str, str]] = []
        self.copies: list[tuple[Path, str]] = []
        self.markers: dict[str, str] = {}
        self.archive = b""
        self.launch_error: Exception | None = None
        self.copy_error: Exception | None = None
        self.collect_error: Exception | None = None

    def ssh(self, command: str, *, timeout=None):
        self.calls.append(("ssh", command))
        if "nohup" in command:
            if self.launch_error is not None:
                raise self.launch_error
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "wc -c" in command and "cat --" in command:
            for name, marker in self.markers.items():
                if name in command:
                    return SimpleNamespace(returncode=0, stdout=marker, stderr="")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def scp(self, source: Path, destination: str, *, timeout=None):
        self.calls.append(("scp", destination))
        self.copies.append((source, destination))
        if self.copy_error is not None:
            raise self.copy_error
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def ssh_bytes(self, command: str, *, timeout=None):
        self.calls.append(("tar", command))
        if self.collect_error is not None:
            raise self.collect_error
        assert command == f"tar -C {ROOT}/out -czf - --exclude=./rollouts/responses ."
        return SimpleNamespace(returncode=0, stdout=self.archive, stderr=b"")


def archive(files: dict[str, bytes], *, special: tuple[str, str] | None = None) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o600
            tar.addfile(info, io.BytesIO(data))
        if special is not None:
            name, target = special
            info = tarfile.TarInfo(name)
            info.type = tarfile.SYMTYPE
            info.linkname = target
            tar.addfile(info)
    return output.getvalue()


def make_remote(tmp_path: Path) -> tuple[ScorerRemote, FakeRemote, Path]:
    key = tmp_path / "id_test"
    tmp_path.mkdir(parents=True, exist_ok=True)
    key.write_text("private-key-contents", encoding="utf-8")
    key.chmod(0o600)
    work = tmp_path / "work"
    job = work / "jobs" / EPOCH / "job.json"
    job.parent.mkdir(parents=True)
    job.write_text(json.dumps({"epoch": EPOCH, "head_version": "a" * 64, "tasks": ["a" * 20]}), encoding="utf-8")
    config = SSHConfig("scorer.internal", 2222, key, work)
    remote = FakeRemote(config)
    return ScorerRemote(config, runner=remote), remote, key


def public_archive() -> bytes:
    return archive({
        "./results.json": b'{"epoch":"' + EPOCH.encode() + b'","scores":[1]}',
        "./manifest.json": b'{"public":true}',
        "./accepted.jsonl": b'{"id":"a"}\n',
        "./tasks.jsonl": b'{"id":"a"}\n',
        "./rollouts/one.jsonl": b'{"sample":1}\n',
    })


def ordinary_archive(files: dict[str, bytes]) -> bytes:
    """Real tar output includes directories and a leading ./ entry."""
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as tar:
        tar.addfile(tarfile.TarInfo("."))
        directories = {str(Path(name).parent) for name in files if str(Path(name).parent) != "."}
        for name in sorted(directories):
            info = tarfile.TarInfo(name)
            info.type = tarfile.DIRTYPE
            tar.addfile(info)
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return output.getvalue()


def test_dispatch_is_fixed_async_private_and_idempotent(tmp_path):
    remote, fake, key = make_remote(tmp_path)
    status = remote.dispatch(EPOCH)
    assert status.state == "dispatched"
    assert fake.copies == [
        (remote.work_dir / "jobs" / EPOCH / "job.json", f"root@scorer.internal:{ROOT}/job.json")
    ]
    prepare, ensure, launch = [command for kind, command in fake.calls if kind == "ssh"]
    assert prepare == f"umask 077 && mkdir -p -- {ROOT}"
    assert "sn120_v2.generator.ensure_head " + "a" * 64 in ensure
    assert "nohup /bin/sh -c " in launch and launch.endswith("&")
    assert "< /dev/null >> " in launch and "2>&1 &" in launch
    assert f"PYTHONPATH=/root/gen/src /root/venv/bin/python -m sn120_v2.generator.scorer run {ROOT}/job.json" in launch
    assert "mv -- \"$tmp\" " + ROOT + "/done.json" in launch
    assert "private-key-contents" not in launch and str(key) not in launch
    before = list(fake.calls)
    assert remote.dispatch(EPOCH).state == "dispatched"
    assert fake.calls == before
    assert (remote.work_dir / "jobs" / EPOCH / ".remote.lock").is_file()


def test_gzip_collection_excludes_private_journal_and_preserves_member_limits(tmp_path):
    remote, fake, _ = make_remote(tmp_path)
    remote.dispatch(EPOCH)
    fake.markers["done.json"] = '{"code":0}'
    fake.archive = public_archive()
    assert remote.collect(EPOCH).complete
    assert any("--exclude=./rollouts/responses" in command for kind, command in fake.calls if kind == "tar")
    assert not (remote.work_dir / "epochs" / EPOCH / "out" / "rollouts" / "responses").exists()
    assert (remote.work_dir / "epochs" / EPOCH / "out" / "rollouts" / "one.jsonl").is_file()
    other, other_fake, _ = make_remote(tmp_path / "unsafe")
    other.dispatch(EPOCH)
    other_fake.markers["done.json"] = '{"code":0}'
    other_fake.archive = archive({"./results.json": b'{"ok":true}', "../escape": b"bad"})
    with pytest.raises(InfraError):
        other.collect(EPOCH)


def test_collect_requires_success_marker_and_preserves_public_tree(tmp_path):
    remote, fake, _ = make_remote(tmp_path)
    fake.archive = public_archive()
    assert remote.collect(EPOCH).state == "absent"
    remote.dispatch(EPOCH)
    assert remote.collect(EPOCH).state == "running"
    assert not (remote.work_dir / "epochs" / EPOCH / "out").exists()
    fake.markers["done.json"] = json.dumps({"epoch": EPOCH, "code": 0})
    status = remote.collect(EPOCH)
    assert status.complete and status.state == "complete"
    assert status.out_dir == remote.work_dir / "epochs" / EPOCH / "out"
    assert status.results_path == status.out_dir / "results.json"
    assert (status.out_dir / "accepted.jsonl").is_file()
    assert (status.out_dir / "tasks.jsonl").is_file()
    assert (status.out_dir / "manifest.json").is_file()
    assert (status.out_dir / "rollouts" / "one.jsonl").read_bytes() == b'{"sample":1}\n'
    before = list(fake.calls)
    assert remote.collect(EPOCH).complete
    assert remote.dispatch(EPOCH).complete
    assert fake.calls == before


def test_collect_accepts_ordinary_tar_directory_entries(tmp_path):
    remote, fake, _ = make_remote(tmp_path)
    remote.dispatch(EPOCH)
    fake.markers["done.json"] = '{"code":0}'
    # The leading directory itself is a valid, ordinary tar directory.
    fake.archive = ordinary_archive({
        "./results.json": b'{"ok":true}',
        "./rollouts/one.jsonl": b'{"rollout":1}\n',
    })
    assert remote.collect(EPOCH).complete
    assert (remote.work_dir / "epochs" / EPOCH / "out" / "rollouts" / "one.jsonl").is_file()


def test_local_timeout_during_launch_never_means_failed_or_relaunches(tmp_path):
    remote, fake, _ = make_remote(tmp_path)
    fake.launch_error = InfraError("local SSH timeout; subprocess killed")
    with pytest.raises(InfraError, match="timeout"):
        remote.dispatch(EPOCH)
    assert remote.status(EPOCH, refresh=False).state == "dispatched"
    launches = [call for call in fake.calls if "nohup" in call[1]]
    assert remote.dispatch(EPOCH).state == "dispatched"
    assert remote.collect(EPOCH).state == "running"
    assert [call for call in fake.calls if "nohup" in call[1]] == launches


def test_copy_timeout_is_resumable_without_another_dispatch(tmp_path):
    remote, fake, _ = make_remote(tmp_path)
    fake.copy_error = InfraError("scp timed out")
    with pytest.raises(InfraError):
        remote.dispatch(EPOCH)
    assert remote.status(EPOCH, refresh=False).state == "dispatched"
    assert not any("nohup" in command for _, command in fake.calls)
    assert remote.dispatch(EPOCH).state == "dispatched"
    assert remote.collect(EPOCH).state == "running"
    assert len(fake.copies) == 1


def test_failed_or_nonzero_done_marker_raises_and_preserves_state(tmp_path):
    remote, fake, _ = make_remote(tmp_path)
    remote.dispatch(EPOCH)
    fake.markers["failed.json"] = '{"code":17}'
    with pytest.raises(InfraError, match="17"):
        remote.collect(EPOCH)
    assert remote.status(EPOCH, refresh=False).state == "failed"
    assert not (remote.work_dir / "epochs" / EPOCH / "out").exists()

    other = make_remote(tmp_path / "other")[0]
    other_fake = other.runner
    other.dispatch(EPOCH)
    other_fake.markers["done.json"] = '{"code":1}'
    with pytest.raises(InfraError, match="nonzero"):
        other.collect(EPOCH)
    assert other.status(EPOCH, refresh=False).state == "failed"


def test_missing_required_results_and_retry_after_collection_timeout(tmp_path):
    remote, fake, _ = make_remote(tmp_path)
    remote.dispatch(EPOCH)
    fake.markers["done.json"] = '{"code":0}'
    fake.archive = archive({"./accepted.jsonl": b"{}\n"})
    with pytest.raises(InfraError, match="results.json"):
        remote.collect(EPOCH)
    assert remote.status(EPOCH, refresh=False).state == "collecting"
    assert not (remote.work_dir / "epochs" / EPOCH / "out").exists()

    fake.archive = public_archive()
    fake.collect_error = InfraError("tar download timed out")
    with pytest.raises(InfraError, match="timed out"):
        remote.collect(EPOCH)
    fake.collect_error = None
    assert remote.collect(EPOCH).complete
    assert not (remote.work_dir / "epochs" / EPOCH / ".out-download").exists()


def test_remote_archive_cannot_escape_or_write_links(tmp_path):
    for bad in ("../escape", "/absolute", "a/../../escape"):
        remote, fake, _ = make_remote(tmp_path / bad.replace("/", "_").replace(".", "_"))
        remote.dispatch(EPOCH)
        fake.markers["done.json"] = '{"code":0}'
        fake.archive = archive({"./results.json": b'{"ok":true}', bad: b"bad"})
        with pytest.raises(InfraError):
            remote.collect(EPOCH)
        assert not (tmp_path / "escape").exists()
    remote, fake, _ = make_remote(tmp_path / "link")
    remote.dispatch(EPOCH)
    fake.markers["done.json"] = '{"code":0}'
    fake.archive = archive({"./results.json": b'{"ok":true}'}, special=("rollouts/link", "../../escape"))
    with pytest.raises(InfraError, match="regular files"):
        remote.collect(EPOCH)
    assert not (tmp_path / "escape").exists()


def test_done_for_other_epoch_or_bad_results_fails_closed(tmp_path):
    remote, fake, _ = make_remote(tmp_path)
    remote.dispatch(EPOCH)
    fake.markers["done.json"] = '{"epoch":"20261120T0400Z","code":0}'
    with pytest.raises(InfraError, match="wrong epoch"):
        remote.collect(EPOCH)


def test_config_json_paths_aliases_and_secret_never_in_repr(tmp_path):
    key = tmp_path / "key"
    key.write_text("do-not-print", encoding="utf-8")
    config_path = tmp_path / "ssh.json"
    config_path.write_text(json.dumps({
        "host": "127.0.0.1", "port": 22, "key_path": "key", "work_dir": "jobs-work",
    }), encoding="utf-8")
    config = SSHConfig.from_file(config_path)
    assert config.identity_file == key
    assert config.work_dir == tmp_path / "jobs-work"
    assert SSHConfig.load(config_path) == config
    assert ScorerRemote.from_config(config_path, runner=FakeRemote(config)).config == config
    inline = ScorerRemote.from_config(
        {"host": "127.0.0.1", "port": 22, "key_path": str(key)}, tmp_path / "jobs-work",
        runner=FakeRemote(config),
    )
    assert inline.config == config
    assert str(key) not in repr(config) and "do-not-print" not in repr(config)


@pytest.mark.parametrize(
    "changes",
    [
        {"port": 0}, {"port": True}, {"port": 65536}, {"host": "-oProxyCommand=bad"},
        {"host": "host name"}, {"user": "bad user"}, {"timeout_s": 0},
        {"timeout_s": float("nan")}, {"timeout_s": float("inf")},
    ],
)
def test_config_rejects_bad_endpoint(tmp_path, changes):
    key = tmp_path / "key"
    key.write_text("secret", encoding="utf-8")
    with pytest.raises(ConfigError):
        SSHConfig.from_dict({"host": "localhost", "port": 22, "key_path": key, "work_dir": tmp_path} | changes)


def test_missing_and_empty_jobs_fail_before_network(tmp_path):
    key = tmp_path / "key"
    key.write_text("secret", encoding="utf-8")
    config = SSHConfig("localhost", 22, key, tmp_path / "work")
    fake = FakeRemote(config)
    remote = ScorerRemote(config, fake)
    with pytest.raises(ConfigError):
        remote.dispatch(EPOCH)
    job = remote.work_dir / "jobs" / EPOCH / "job.json"
    job.parent.mkdir(parents=True, exist_ok=True)
    for value in ("", "null", "{}", "[]", '"not an object"'):
        job.write_text(value, encoding="utf-8")
        with pytest.raises(ConfigError):
            remote.dispatch(EPOCH)
    assert fake.calls == []


@pytest.mark.parametrize("epoch", ["../20261120T0000Z", "20261120T0000Z;touch /tmp/pwned", "20261120T0000Z "])
def test_epoch_cannot_become_a_remote_shell_command(tmp_path, epoch):
    remote, fake, _ = make_remote(tmp_path)
    with pytest.raises(ValueError):
        remote.dispatch(epoch)
    with pytest.raises(ValueError):
        remote.collect(epoch)
    assert fake.calls == []


def test_real_runner_argv_has_no_shell_and_strict_bounded_ssh(tmp_path):
    _, _, key = make_remote(tmp_path)
    config = SSHConfig("host", 2222, key, tmp_path / "work", timeout_s=12)
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    runner = SSHRunner(config, run=fake_run)
    runner.ssh("mkdir -p -- /root/gen/jobs/epoch")
    runner.scp(tmp_path / "local job", f"root@host:{ROOT}/job.json")
    ssh, scp = calls
    assert ssh[0][0] == "ssh" and scp[0][0] == "scp"
    for argv, kwargs in calls:
        assert "shell" not in kwargs
        assert kwargs["capture_output"] and kwargs["text"] and kwargs["check"] is False
        assert kwargs["timeout"] == 12
        assert argv[argv.index("-i") + 1] == str(key)
        assert argv[argv.index("-F") + 1] == "/dev/null"
        assert "BatchMode=yes" in argv and "StrictHostKeyChecking=yes" in argv
        assert "ConnectTimeout=12" in argv and "IdentitiesOnly=yes" in argv
        assert argv[argv.index("--") + 1] not in ("-oProxyCommand=bad",)
    assert ssh[0][ssh[0].index("-p") + 1] == "2222"
    assert scp[0][scp[0].index("-P") + 1] == "2222"
    assert scp[0][scp[0].index("--") + 1] == str(tmp_path / "local job")
    assert scp[0][-1] == f"root@host:{ROOT}/job.json"


def test_real_runner_kills_local_timeout_semantics_and_does_not_leak_key(tmp_path):
    _, _, key = make_remote(tmp_path)
    config = SSHConfig("host", 22, key, tmp_path / "work")

    def timed_out(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    runner = SSHRunner(config, run=timed_out)
    with pytest.raises(InfraError) as caught:
        runner.ssh("fixed command")
    assert "TimeoutExpired" in str(caught.value)
    assert str(key) not in str(caught.value) and "private-key-contents" not in str(caught.value)


def test_real_sh_launch_writes_marker_only_after_fixed_scorer_success(tmp_path):
    """Execute the fixed shell wrapper with a harmless stand-in for the pod."""
    for code in (0, 9):
        root = tmp_path / f"pod-{code}"
        job = root / "jobs" / EPOCH
        job.mkdir(parents=True)
        (job / "job.json").write_text('{"ok":true}', encoding="utf-8")
        inner = (
            f"/bin/sh -c {shlex.quote(f'exit {code}')}; code=$?; "
            f"if [ \"$code\" -eq 0 ]; then kind=done; else kind=failed; fi; "
            f"tmp={job}/.$kind.$$.json; "
            f"printf '%s\\n' \"{{\\\"epoch\\\":\\\"{EPOCH}\\\",\\\"code\\\":$code}}\" > \"$tmp\" && "
            f"if [ \"$code\" -eq 0 ]; then mv -- \"$tmp\" {job}/done.json; "
            f"else mv -- \"$tmp\" {job}/failed.json; fi; exit \"$code\""
        )
        launch = f"{{ nohup /bin/sh -c {shlex.quote(inner)} ; }} < /dev/null >> {job}/scorer.log 2>&1 &"
        # This is a test of the remote-side shell only, not a subprocess used
        # by the implementation.  It must return without waiting for nohup.
        subprocess.run(["/bin/sh", "-c", launch], timeout=3, check=True)
        for _ in range(100):
            if (job / "done.json").exists() or (job / "failed.json").exists():
                break
            import time
            time.sleep(0.01)
        marker = json.loads((job / ("done.json" if code == 0 else "failed.json")).read_text())
        assert marker == {"epoch": EPOCH, "code": code}
        assert (job / "done.json").exists() is (code == 0)
