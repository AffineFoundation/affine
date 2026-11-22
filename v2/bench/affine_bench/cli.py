"""affine-bench CLI.
  node-run   (node) evaluate one model on benchmarks          affine-bench node-run --help
  daemon     (host) the service: starts a worker for each row that has work, and writes the table
  row        (host) the worker of one row; the service starts it   affine-bench row MODEL_ID
  once       (host) benchmark every waiting row, give every pod back, exit
These read only:
  status     (host) the rows in work, the queue, the rows that wait
  offers     (host) the Lium offers that a rent could take now: kind, price, count
  pods       (host) every bench pod at Lium with its row; a pod that no row claims; a row that has no pod
  retry      (host) queue a stopped row again (no stop of the service)   affine-bench retry MODEL_ID...
The next two need the service stopped (one service per state directory):
  publish    (host) rebuild benchmarks.json now and write it to the store when it changed
  release    (host) give the pod of a row back now             affine-bench release MODEL_ID..."""
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
