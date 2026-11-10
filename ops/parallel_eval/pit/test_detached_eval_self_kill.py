"""detached_eval_command must not TERM itself via pgrep on its own cmdline.

Repro class (e1067): the setsid/nohup bash -c wrapper embeds 'vllm serve' /
'podctl.py serve' in argv; a literal pgrep -f of those patterns kills $$ .
"""
from __future__ import annotations

import os
import stat
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


class TestDetachedEvalSelfKill(unittest.TestCase):
    def test_generated_shell_survives_fake_pgrep_kill(self):
        from ops.parallel_eval.live.pe_box import detached_eval_command

        cmd = detached_eval_command(
            job_dir='/workspace/pst/pit-jobs/e1067',
            candidate_id='c' * 64,
            plan_path='/workspace/pst/pit-jobs/e1067/eval-plan.json',
        )
        # Must not use the self-killing literal pgrep pattern.
        self.assertNotIn(
            "pgrep -f 'train_pit.py|v83_dp.py|vllm serve|podctl.py serve'",
            cmd,
        )
        self.assertIn('[v]llm serve', cmd)
        self.assertIn('[t]rain_pit', cmd)
        self.assertIn('query-compute-apps=pid', cmd)
        self.assertIn('_kill_one', cmd)
        self.assertIn('MYPGID', cmd)
        self.assertIn('job.pid', cmd)

        # Execute only the shell preamble (stop before python3 heredoc) under a
        # fake PATH where pgrep reports our own pid and kill records targets.
        preamble = cmd.split('python3 - ')[0]
        self.assertTrue(preamble.strip())

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            bin_dir = td_path / 'bin'
            bin_dir.mkdir()
            job = td_path / 'job'
            job.mkdir()
            (job / 'markers').mkdir()
            (job / 'job.pid').write_text('999001\n')
            kill_log = td_path / 'kills.log'
            kill_log.write_text('')

            # Fake pgrep: always print the caller's PID (would self-kill with old code).
            pgrep = bin_dir / 'pgrep'
            pgrep.write_text(textwrap.dedent('''\
                #!/bin/bash
                # Report $$ of the *parent* shell that invoked us (the wrapper).
                echo "$PPID"
                exit 0
            '''))
            pgrep.chmod(pgrep.stat().st_mode | stat.S_IEXEC)

            kill = bin_dir / 'kill'
            kill.write_text(textwrap.dedent('''\
                #!/bin/bash
                echo "$*" >> "%s"
                # Never actually signal anyone.
                exit 0
            ''' % kill_log))
            kill.chmod(kill.stat().st_mode | stat.S_IEXEC)

            # Minimal fakes so nvidia-smi / ps / head do not fail the preamble.
            (bin_dir / 'nvidia-smi').write_text(textwrap.dedent('''\
                #!/bin/bash
                # No GPU holders in the hermetic fixture.
                exit 0
            '''))
            (bin_dir / 'nvidia-smi').chmod(
                (bin_dir / 'nvidia-smi').stat().st_mode | stat.S_IEXEC
            )

            # Rewrite job dir placeholder and run preamble; assert we stay alive.
            script = preamble.replace('__JD__', str(job))
            script += '\necho SURVIVED:$$\n'
            runner = td_path / 'run.sh'
            runner.write_text('#!/bin/bash\nset +e\n' + script)
            runner.chmod(runner.stat().st_mode | stat.S_IEXEC)

            env = dict(os.environ)
            env['PATH'] = '%s:%s' % (bin_dir, env.get('PATH', ''))
            proc = subprocess.run(
                ['bash', str(runner)],
                capture_output=True, text=True, timeout=15, env=env,
            )
            out = (proc.stdout or '') + (proc.stderr or '')
            self.assertIn('SURVIVED:', out, msg='wrapper died before echo; out=%r' % out)
            # Kill must never have targeted the surviving shell itself.
            survived_pid = None
            for line in out.splitlines():
                if line.startswith('SURVIVED:'):
                    survived_pid = line.split(':', 1)[1].strip()
            self.assertTrue(survived_pid and survived_pid.isdigit())
            kills = kill_log.read_text()
            # Prior job.pid (999001) may be targeted; our own pid must not.
            self.assertNotIn(survived_pid, kills.split())
            # Negative-group kill of self also forbidden.
            self.assertNotIn('-%s' % survived_pid, kills)

    def test_per_attempt_logs_not_truncated(self):
        from ops.parallel_eval.pit.box_job import (
            MemBoxJobStore, BoxJobController, log_attempt_path, LOG_NAME,
        )
        store = MemBoxJobStore()
        ctrl = BoxJobController(
            store=store, item_id='e1067-logs',
            sleep_fn=lambda _s: None, now_fn=lambda: 0.0,
        )
        ctrl.ensure_dir()
        p0 = store.launch_detached(ctrl.job_dir, stage='train', command='train_pit.py')
        store.alive_pids.discard(p0)
        store.pending.pop(p0, None)
        # Stamp attempt-0 content that must survive the next launch.
        p0_log = log_attempt_path(ctrl.job_dir, 0)
        store.write_text(p0_log, 'ATTEMPT0-UNIQUE\n')
        p1 = store.launch_detached(ctrl.job_dir, stage='eval', command='eval')
        self.assertIsNotNone(store.read_text(p0_log))
        self.assertIn('ATTEMPT0-UNIQUE', store.read_text(p0_log) or '')
        p1_log = log_attempt_path(ctrl.job_dir, 1)
        self.assertIsNotNone(store.read_text(p1_log))
        # job.log points at latest attempt content.
        latest = store.read_text('%s/%s' % (ctrl.job_dir, LOG_NAME))
        self.assertIsNotNone(latest)
        self.assertIn('eval', latest or '')


if __name__ == '__main__':
    unittest.main()
