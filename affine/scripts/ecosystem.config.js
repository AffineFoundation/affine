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
  ],
};
