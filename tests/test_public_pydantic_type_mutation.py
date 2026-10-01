import unittest,json
from unittest.mock import patch
from subnet.public_pydantic_type_mutation import proposals
class Tests(unittest.TestCase):
 def test_preserves_public_key_order_and_changes_only_type(self):
  public='```json\n{"name":"x","age":1}\n```'
  with patch('subnet.public_pydantic_type_mutation.original_proposals',return_value=[public,'unused']):
   pair=proposals([{'role':'user','content':'public'}])
  self.assertEqual(pair[0],public);self.assertEqual(pair[1],'```json\n{"name":[],"age":1}\n```')
 def test_real_native_type_validation(self):
  from pydantic import BaseModel,ValidationError
  class Schema(BaseModel):name:str;age:int
  with patch('subnet.public_pydantic_type_mutation.original_proposals',return_value=['```json\n{"name":"x","age":1}\n```','unused']): pair=proposals([])
  Schema.model_validate_json(pair[0][8:-4])
  with self.assertRaises(ValidationError):Schema.model_validate_json(pair[1][8:-4])
 def test_unsupported_empty_root_fails(self):
  with patch('subnet.public_pydantic_type_mutation.original_proposals',return_value=['```json\n{}\n```','unused']):
   with self.assertRaises(ValueError):proposals([])
if __name__=='__main__':unittest.main()
