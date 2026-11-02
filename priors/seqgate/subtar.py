"""stream a subset of a tar (stdin) to stdout as a tar: members whose name starts with one of argv[1:]"""
import sys, tarfile
pre = tuple(sys.argv[1:])
src = tarfile.open(fileobj=sys.stdin.buffer, mode='r|'); dst = tarfile.open(fileobj=sys.stdout.buffer, mode='w|', format=tarfile.GNU_FORMAT)
for m in src:
    if not m.name.lstrip('./').startswith(pre): continue
    dst.addfile(m, src.extractfile(m) if m.isfile() else None)
dst.close()
