// pm2 process file for the root machine. The validator embeds the
// provisioner, bench orchestrator, and dashboard pusher, so one process is
// the whole control plane.
//
// Secrets MUST come from doppler on every start/restart. Wrapping the
// interpreter in `doppler run` means `pm2 restart` keeps AFFINE_EVAL_TOKEN /
// HF_TOKEN — a bare `pm2 restart` previously dropped them and the provisioner
// 401'd both pods into a terminate/re-rent loop.
module.exports = {
  apps: [
    {
      name: "affine-validator",
      cwd: __dirname + "/..",
      script: "doppler",
      args: "run -- ../.venv/bin/python -m affine.validator",
      interpreter: "none",
      autorestart: true,
      max_restarts: 1000,
      restart_delay: 10000,
      kill_timeout: 30000,
      out_file: "logs/validator.out.log",
      error_file: "logs/validator.err.log",
      merge_logs: true,
      env: {
        PYTHONUNBUFFERED: "1",
      },
    },
    // Read-only hot-path dashboard API + static site. Separate from the
    // validator so public traffic never blocks weight-setting / duels.
    {
      name: "affine-dash",
      cwd: __dirname + "/..",
      script: "doppler",
      args: "run -- ../.venv/bin/python -m affine.dash",
      interpreter: "none",
      autorestart: true,
      max_restarts: 1000,
      restart_delay: 5000,
      kill_timeout: 10000,
      out_file: "logs/dash.out.log",
      error_file: "logs/dash.err.log",
      merge_logs: true,
      env: {
        PYTHONUNBUFFERED: "1",
      },
    },
    // TLS edge on :8443 (443 is owned by forest-head / arbos.life).
    {
      name: "affine-caddy",
      cwd: __dirname + "/..",
      script: "caddy",
      args: "run --config Caddyfile",
      interpreter: "none",
      autorestart: true,
      max_restarts: 1000,
      restart_delay: 5000,
      out_file: "logs/caddy.out.log",
      error_file: "logs/caddy.err.log",
      merge_logs: true,
    },
    // Daily datagen → production corpus refresh at 16:00 UTC (host is UTC).
    // Runs once per cron fire and exits (autorestart off). Deliberately NOT
    // wrapped in doppler (local doppler token is broken): the script sources
    // Hippius keys from the running validator's process env and caches the
    // datagen HF token itself — see ops/datagen_refresh.py.
    {
      name: "affine-corpus-refresh",
      cwd: __dirname + "/../..",
      script: ".venv/bin/python",
      args: "ops/datagen_refresh.py",
      interpreter: "none",
      autorestart: false,
      cron_restart: "0 16 * * *",
      out_file: "affine/logs/corpus_refresh.out.log",
      error_file: "affine/logs/corpus_refresh.err.log",
      merge_logs: true,
      env: {
        PYTHONUNBUFFERED: "1",
      },
    },
    // Legacy public hostname https://sn120.arbos.life → 127.0.0.1:8787.
    // The official site https://affine.io reaches 127.0.0.1:8787 via
    // Cloudflare (proxied DNS) → system caddy :80, not this tunnel.
    // Token: AFFINE_CF_TUNNEL_TOKEN in doppler (arbos/dev).
    {
      name: "affine-tunnel",
      cwd: __dirname + "/..",
      script: "doppler",
      args: "run -- bash -lc 'exec cloudflared tunnel --no-autoupdate run --token \"$AFFINE_CF_TUNNEL_TOKEN\"'",
      interpreter: "none",
      autorestart: true,
      max_restarts: 1000,
      restart_delay: 5000,
      out_file: "logs/tunnel.out.log",
      error_file: "logs/tunnel.err.log",
      merge_logs: true,
    },
  ],
};
