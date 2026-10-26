from pathlib import Path
from types import SimpleNamespace
import pytest
from ops.prepare_mimo_worker_storage import prepare


def test_fast_volume_counts_both_generations_and_one_cache(tmp_path):
    mounts=tmp_path/'mounts';mounts.write_text('overlay / overlay rw 0 0\n')
    result=prepare(tmp_path/'worker',generation_bytes=100,cache_bytes=20,reserve_bytes=5,
        mounts=mounts,disk_usage=lambda _:SimpleNamespace(free=225))
    assert result['required_free_bytes']==225
    assert Path(result['trainer_state']).is_dir() and result['live_state_moved']is False
    with pytest.raises(ValueError,match='two-generation'):
        prepare(tmp_path/'too-small',generation_bytes=100,cache_bytes=20,reserve_bytes=5,
            mounts=mounts,disk_usage=lambda _:SimpleNamespace(free=224))
    assert not (tmp_path/'too-small').exists()


def test_fuse_and_symlink_cannot_be_admitted(tmp_path):
    mounts=tmp_path/'mounts';mounts.write_text(f'overlay / overlay rw 0 0\nroot {tmp_path} fuse.gocryptfs rw 0 0\n')
    with pytest.raises(ValueError,match='fast disk'):
        prepare(tmp_path/'slow',mounts=mounts,disk_usage=lambda _:SimpleNamespace(free=10**15))
    (tmp_path/'link').symlink_to(tmp_path,target_is_directory=True)
    with pytest.raises(ValueError,match='ancestry'):prepare(tmp_path/'link'/'new',mounts=mounts)
