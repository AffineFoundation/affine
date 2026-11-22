"""ssh to the bench node through the wrapper script (BatchMode, pinned known_hosts)."""
import os, subprocess
from . import config as C


class Node:
    def __init__(self, ssh=C.NODE_SSH):
        self.ssh = ssh

    def run(self, cmd, inp=None, timeout=120):
        return subprocess.run([self.ssh, cmd], input=inp, capture_output=True, text=True, timeout=timeout)

    def save(self, cmd, path, timeout=900):
        """What a command on the node writes to its output, as bytes into a private file on the host. True when the
        command ended well and wrote something; else the file is not made."""
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            os.chmod(tmp, 0o600)
            p = subprocess.run([self.ssh, cmd], stdout=f, stderr=subprocess.DEVNULL, timeout=timeout)
        if p.returncode != 0 or not os.path.getsize(tmp):
            os.remove(tmp); return False
        os.replace(tmp, path); return True

    def out(self, cmd, timeout=120):
        try:
            p = self.run(cmd, timeout=timeout)
        except subprocess.TimeoutExpired:
            return ""
        return p.stdout
