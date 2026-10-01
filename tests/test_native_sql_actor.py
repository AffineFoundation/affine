import hashlib
import json
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from subnet.native_sql_actor import public_descriptor,PublicSQLActor,command,REVISION,RPC,START
from subnet.native_sql_isolation import BASE

class PublicSQLBoundary(unittest.TestCase):
    def test_only_public_database_schema_question_emitted(self):
        with TemporaryDirectory() as directory:
            db=Path(directory)/'public.sqlite';connection=sqlite3.connect(db)
            connection.execute('CREATE TABLE visible (value INTEGER)');connection.close()
            private={'db_id':'public','db_path':str(db),'database_sha256':hashlib.sha256(db.read_bytes()).hexdigest(),'question':'How many visible rows?','gold_sql':'OPERATOR_REFERENCE_SENTINEL'}
            public=public_descriptor(private);text=json.dumps(public)
            self.assertIn('CREATE TABLE visible',text);self.assertIn('How many visible rows?',text)
            self.assertNotIn('OPERATOR_REFERENCE_SENTINEL',text);self.assertNotIn('db_path',text);self.assertNotIn('gold_sql',text)
    def test_miner_cannot_add_private_fields_to_actor_descriptor(self):
        public=dict(revision=REVISION,db_id='public',database_sha256='a',gold_sql='private')
        with self.assertRaisesRegex(ValueError,'descriptor fields'):PublicSQLActor({'db_id':'public','database_sha256':'a'},public)
    def test_public_runtime_cannot_select_mutable_image(self):
        with self.assertRaises(ValueError):command('affine-sql-public-'+'a'*16,{'revision':REVISION,'base_image':BASE,'image':'python:latest'})
    def test_invalid_action_cannot_execute_or_mutate_actor(self):
        public={'revision':REVISION,'db_id':'public','database_sha256':'a','original_source_sha256':'b','messages':[],'tools':[]}
        actor=PublicSQLActor({'db_id':'public','database_sha256':'a'},public);actor.started=True
        for name,args in [('operator',{'command':'SELECT 1'}),('bash',{'command':1}),('bash',{'command':'a'*16385}),('bash',{'command':'ls','host_path':'/'})]:
            with self.subTest(name=name,args_type=type(args['command'])),self.assertRaises(ValueError):actor.call(name,args)
