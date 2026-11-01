"""affine-bench CLI.
  node-run   (node) evaluate one model on benchmarks          affine-bench node-run --help
  daemon     (host) queue-driven scheduler + publisher        affine-bench daemon
  run        (host) one model x benchmark(s) through the node  affine-bench run --model king:<hash>|<ref-id> --benchmarks math500
  publish    (host) rebuild affine.io/benchmarks.json now
  status     (host) queue / results summary"""
import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__); return
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "node-run":
        from .node_run import main as m; return m(rest)
    from . import daemon
    return daemon.cli(cmd, rest)


if __name__ == "__main__":
    main()
