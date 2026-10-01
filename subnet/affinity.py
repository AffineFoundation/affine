"""Per-process CPU count bound, portable across different CPU ID maps."""
import argparse
import os


def main():
    p=argparse.ArgumentParser();p.add_argument('--cores',type=int,default=4);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    if not 1<=a.cores<=64 or not a.command:raise ValueError('CPU count/command')
    allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(allowed[:a.cores]))
    command=a.command[1:] if a.command[0]=='--' else a.command
    os.execvp(command[0],command)

if __name__=='__main__':main()
