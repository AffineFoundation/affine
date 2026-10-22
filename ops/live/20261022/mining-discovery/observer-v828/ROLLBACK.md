# Rollback: mining.json observer rebind (20261021)
Effect: returns observer to the v8.2.4 binding (which REFUSES the running v8.2.8 lifecycle -> discovery_binding_error).
Only roll back if the lifecycle itself is rolled back to v8.2.4, or the new observer misbehaves.

    rm "/home/const/.config/systemd/user/affine-mining-discovery.service.d/zzzz~~~~~mimo-observer-v828.conf"
    systemctl --user daemon-reload
    systemctl --user restart affine-mining-discovery.service
    journalctl --user -u affine-mining-discovery.service -n 5 --no-pager

Full restore of the drop-in dir: backup/affine-mining-discovery.service.d.tar (tar -C ~/.config/systemd/user -xf ...).
Prior public document: backup/mining.json.before. The observer only writes the local static mining.json; no R2/chain state to undo.
If v8.2.8 is rolled back to v8.2.7 or v8.2.6, rebind instead: set host_capture_config/host_uid_config (+files dirs) to that
release, drop host_v828_config (the patched observer accepts capture v1/v2/v3 directly).
