"""affine-bench CLI.
  node-run   (node) evaluate one model on benchmarks          affine-bench node-run --help
  daemon     (host) queue-driven scheduler + publisher        affine-bench daemon
  once       (host) benchmark every waiting king, release the pod, exit
  publish    (host) rebuild benchmarks.json now
  status     (host) queue / results summary"""
import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__); return
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "node-run":
        from .node_run import main as m; return m(rest)
    from . import daemon  # imported here, not at the top: the host modules need the host environment, the node has none
    return daemon.cli(cmd, rest)


if __name__ == "__main__":
    main()
