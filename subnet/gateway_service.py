"""Retain an existing private capability gateway and public frozen audit history."""
import argparse,json,time
from pathlib import Path
from .storage import Bucket,Gateway

def main():
    p=argparse.ArgumentParser();p.add_argument('--bucket-config',required=True);p.add_argument('--state-path',required=True);p.add_argument('--public-url',required=True);p.add_argument('--port',type=int,required=True);a=p.parse_args()
    gateway=Gateway(Bucket(json.loads(Path(a.bucket_config).read_text())),port=a.port,state_path=a.state_path,public_url=a.public_url)
    try:
        while True:time.sleep(30)
    finally:gateway.stop()
if __name__=='__main__':main()
