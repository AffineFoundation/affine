"""Apply ROOT-signed explicit evaluator inventory using normal retention software."""
import argparse,json
from pathlib import Path
from subnet.evaluator_cache_lifecycle import adopt

def main():
    p=argparse.ArgumentParser();p.add_argument('--catalog',required=True);p.add_argument('--authority',required=True);a=p.parse_args()
    print(json.dumps(adopt(json.loads(Path(a.catalog).read_bytes()),a.authority)))
if __name__=='__main__':main()
