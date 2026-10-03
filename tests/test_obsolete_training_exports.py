import base64,hashlib,json,os,sqlite3,tempfile,unittest
from pathlib import Path
from nacl.signing import SigningKey
from subnet.storage import canonical
from ops.retain_completed_training import sha
from ops.retain_obsolete_training_exports import final_candidate,protected_checkpoints

class ObsoleteExportControls(unittest.TestCase):
    def test_current_successor_and_unfinished_original_jobs_are_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);state=root/'state';roles=state/'roles';roles.mkdir(parents=True)
            (state/'controller.json').write_bytes(canonical({'checkpoint':{'id':'a'*64},'active':{'next_checkpoint':{'id':'b'*64}}}))
            config=root/'config.json';config.write_bytes(canonical({'state':str(state)}));record=root/'process.json';ticks=Path('/proc',str(os.getpid()),'stat').read_text().rsplit(')',1)[1].split()[19];record.write_bytes(canonical({'child_pid':os.getpid(),'child_ticks':ticks,'config_sha256':sha(config)}))
            key=SigningKey.generate();authority=key.verify_key.encode().hex()
            def sign(value):return {'payload':value,'signer':authority,'signature':base64.b64encode(key.sign(canonical(value)).signature).decode()}
            with sqlite3.connect(roles/'verifier-queue.sqlite3') as db:
                db.execute('create table jobs(envelope text,status text)')
                for cp,status in [('c'*64,'leased'),('d'*64,'queued'),('e'*64,'complete')]:
                    db.execute('insert into jobs values(?,?)',(json.dumps(sign({'manifest':sign({'checkpoint':{'id':cp}})})),status))
            self.assertEqual(protected_checkpoints(config,record,authority)[1],{'a'*64,'b'*64,'c'*64,'d'*64})
            with sqlite3.connect(roles/'verifier-queue.sqlite3') as db:db.execute('update jobs set envelope=? where status="queued"',(json.dumps(sign({'manifest':sign({'checkpoint':{'id':'changed'}})})).replace('"signature": "','"signature": "AAAA'),))
            with self.assertRaises(Exception):protected_checkpoints(config,record,authority)

    def test_export_requires_complete_original_file_map_and_exact_step_path(self):
        files={'config.json':'a'*64,'model.safetensors':'b'*64};cp=hashlib.sha256(canonical(files)).hexdigest();job={'job_id':'job-original','steps':3};report={'new_checkpoint':{'id':cp,'files':files,'path':'/root/trainer/jobs/job-original/checkpoint-step-3'}}
        self.assertEqual(final_candidate(job,report)['workspace'],'/root/trainer')
        for update in [{'id':'0'*64},{'path':'/root/trainer/jobs/other/checkpoint-step-3'},{'path':'/root/trainer/jobs/job-original/checkpoint-step-2'},{'files':{'config.json':'a'*64}}]:
            with self.subTest(update=update),self.assertRaises(ValueError):final_candidate(job,{'new_checkpoint':dict(report['new_checkpoint'],**update)})

if __name__=='__main__':unittest.main()
