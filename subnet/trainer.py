"""Trainer execution role: accepts only controller-authorized verified pairs."""
import argparse
import json
from pathlib import Path
from .model import Runtime,check_runtime_profile

def main():
    p=argparse.ArgumentParser();p.add_argument('job');p.add_argument('metrics');a=p.parse_args()
    job=json.loads(Path(a.job).read_text())
    check_runtime_profile(job)
    runtime=Runtime(job['checkpoint'],job['files'],environment=job.get('environment'),harness=job.get('harness'))
    metrics=runtime.train(job['pairs'],job['destination'],job['steps'])
    Path(a.metrics).write_text(json.dumps(metrics))
if __name__=='__main__':main()
