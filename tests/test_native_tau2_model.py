import copy,json,os,unittest
from unittest.mock import patch
from nacl.exceptions import BadSignatureError
from nacl.signing import SigningKey
from subnet.native_tau2_model import PROFILE,authenticate,canonical,envelope,profile,render,derived_response_message,validate_derived_response

class RecordingTokenizer:
    def apply_chat_template(self,messages,**kwargs):self.messages=messages;return [1,2]

class NativeModelTests(unittest.TestCase):
    def test_all_original_schema_fields_and_messages_preserved(self):
        tokenizer=RecordingTokenizer();tools=[{'type':'function','function':{'name':'native','description':'Original lengthy policy','parameters':{'type':'object','properties':{'id':{'type':'string','description':'Keep this condition'}},'required':['id']}}}]
        request={'model':'native-model-user','messages':[{'role':'system','content':'Complete original system policy'},{'role':'tool','content':'Original observation','tool_call_id':'x'}],'tools':tools,'temperature':.7}
        self.assertEqual(render(tokenizer,request),[1,2])
        system=tokenizer.messages[0]['content'];self.assertIn('Complete original system policy',system);self.assertIn(canonical(tools).decode(),system)
        self.assertIn('Original observation',tokenizer.messages[1]['content']);self.assertIn('"tool_call_id":"x"',tokenizer.messages[1]['content'])
        self.assertEqual(tokenizer.messages[1]['role'],'user')
    def test_user_observation_and_reward_tampering_rejected_by_receipt_authority(self):
        key=SigningKey.generate();authority=key.verify_key.encode().hex()
        for original,field in [({'role':'user','text':'Model-generated customer observation'},'text'),({'reward':0.,'termination':'max_steps'},'reward')]:
            receipt=envelope(original,key);self.assertEqual(authenticate(receipt,authority),original)
            mutated=copy.deepcopy(receipt);mutated['payload'][field]='fake'
            with self.assertRaises(BadSignatureError):authenticate(mutated,authority)
    def test_wrong_receipt_authority_rejected(self):
        one=SigningKey.generate();two=SigningKey.generate();receipt=envelope({'seed':3,'role':'agent'},one)
        with self.assertRaisesRegex(ValueError,'authority'):authenticate(receipt,two.verify_key.encode().hex())
    def record(self,text,number=2):
        message,finish=derived_response_message(text,number)
        return {'text':text,'request':{'model':'native-model-user'},'prompt':[1,2],'output':[3],'created_at':100.,'completed_at':101.,'response':{'id':f'native-{number}','object':'chat.completion','created':101,'model':'native-model-user','choices':[{'index':0,'message':message,'finish_reason':finish}],'usage':{'prompt_tokens':2,'completion_tokens':1,'total_tokens':3}}}
    def test_last_user_response_must_match_verified_decoded_tokens(self):
        record=self.record('Actual user-model output');validate_derived_response(record,2)
        record['response']['choices'][0]['message']['content']='Forged last user observation'
        with self.assertRaisesRegex(ValueError,'derived response'):validate_derived_response(record,2)
    def test_tool_calls_must_match_exact_output_action_parser(self):
        record=self.record('{"tool_call":{"name":"real_tool","arguments":{"id":"one"}}}')
        self.assertIsNone(record['response']['choices'][0]['message']['content']);validate_derived_response(record,2)
        record['response']['choices'][0]['message']['tool_calls'][0]['function']['name']='forged_tool'
        with self.assertRaisesRegex(ValueError,'derived response'):validate_derived_response(record,2)
    def test_response_metadata_and_usage_are_bound(self):
        for field,value in [('id','other'),('model','native-model-agent'),('created',999),('usage',{'prompt_tokens':0,'completion_tokens':0,'total_tokens':0})]:
            record=self.record('actual');record['response'][field]=value
            with self.assertRaises(ValueError):validate_derived_response(record,2)
    def test_cpu_profile_cannot_be_relaxed(self):
        with patch.dict(os.environ,PROFILE,clear=False):
            profile()
            with patch.dict(os.environ,{'ATEN_CPU_CAPABILITY':'avx2'},clear=False):
                with self.assertRaisesRegex(ValueError,'strict CPU'):profile()

if __name__=='__main__':unittest.main()
