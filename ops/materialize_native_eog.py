"""Materialize operator-private original fixtures from the existing pinned cache."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from datasets import Dataset
from subnet.native_eog_isolation import original_modules, IMAGE

REVISION='c8e538eae8a6205294f0a86675fefdc1fac408f6'
ARCHIVE='https://raw.githubusercontent.com/ServiceNow/EnterpriseOps-Gym/de22905d21a080b83bf4a54258afe4250ee2dd55/gym_dbs.zip'

def main():
    p=argparse.ArgumentParser();p.add_argument('--state',type=Path,default=Path('state/native-eog-isolation'))
    p.add_argument('--cache',type=Path,default=Path.home()/'.cache/huggingface/datasets');args=p.parse_args()
    cache=args.cache;state=args.state;state.mkdir(parents=True,exist_ok=True)
    arrow=cache/'ServiceNow-AI___enterprise_ops-gym/oracle/0.0.0'/REVISION/'enterprise_ops-gym-calendar.arrow'
    dataset=Dataset.from_file(str(arrow));archive=None
    for metadata in (cache/'downloads').glob('*.json'):
        try:value=json.loads(metadata.read_text())
        except (ValueError,OSError):continue
        if value.get('url')==ARCHIVE:
            archive=metadata.with_suffix('');break
    if archive is None or not archive.is_file():raise FileNotFoundError('original pinned database archive cache')
    selected=[0,1,2,3,54];rows=[];inventory=[];taskset,_,wrapper=original_modules()
    with zipfile.ZipFile(archive) as zipped:
        for index in selected:
            raw=dataset[index];seed=json.loads(raw['gym_servers_config'])[0]['seed_database_file']
            matches=[root for root in (cache/'downloads/extracted').iterdir() if (root/seed).is_file()]
            if len(matches)!=1:raise ValueError('ambiguous/missing original seed extraction')
            dbroot=matches[0];actual=(dbroot/seed).read_bytes()
            members=[name for name in zipped.namelist() if name==seed or name.endswith('/'+seed)]
            if len(members)!=1 or zipped.read(members[0])!=actual:raise ValueError('original seed archive/extraction mismatch')
            data=taskset._parse_row(index,raw,dbroot)
            # Match the original Affine wrapper's load transformation exactly.
            system=data.system_prompt or ''
            if wrapper.TOOL_MARKER not in system.lower():
                data=data.model_copy(update={'system_prompt':system.rstrip()+wrapper.TOOL_NOTE})
            if len(data.services)!=1 or data.services[0].image!=IMAGE:raise ValueError('original Calendar service definition')
            rows.append({'data':data.model_dump(mode='json'),'task_config':taskset.config.task.model_dump(mode='json')})
            inventory.append({'index':index,'task_id':data.name,'seed_bytes':len(actual),'seed_sha256':hashlib.sha256(actual).hexdigest()})
    for name,value in [('original-calendar-four.private.json',rows[:4]),('original-calendar-relocation.private.json',rows[4])]:
        path=state/name;path.write_text(json.dumps(value));path.chmod(0o600)
    report={'dataset':'ServiceNow-AI/EnterpriseOps-Gym','revision':REVISION,'split':'calendar','rows':len(dataset),
        'arrow_sha256':hashlib.sha256(arrow.read_bytes()).hexdigest(),'archive_url':ARCHIVE,
        'archive_bytes':archive.stat().st_size,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
        'original_seeds_exactly_match_archive':True,'fixtures':inventory}
    (state/'fixture-materialization.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'fixtures':len(rows),'original_seeds_exactly_match_archive':True}))

if __name__=='__main__':main()
