"""ssh to the bench node through the wrapper script (BatchMode, pinned known_hosts)."""
import subprocess
from . import config as C


class Node:
    def __init__(self, ssh=C.NODE_SSH):
        self.ssh = ssh

    def run(self, cmd, inp=None, timeout=120):
        return subprocess.run([self.ssh, cmd], input=inp, capture_output=True, text=True, timeout=timeout)

    def out(self, cmd, timeout=120):
        p = self.run(cmd, timeout=timeout)
        return p.stdout
