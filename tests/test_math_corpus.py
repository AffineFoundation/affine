import unittest
from subnet.math_corpus import adapt_row,public_messages,question_key,last_boxed_body,snapshot_row

class MathCorpusTests(unittest.TestCase):
 def test_separate_numina_cot_and_no_solution_prompt(self):
  r=adapt_row('NuminaMath-CoT',{'problem':'Compute 1+1.','solution':'PRIVATE reasoning \\boxed{\\frac{4}{2}}','source':'a'},3)
  self.assertEqual(r['reference'],'\\frac{4}{2}');self.assertNotIn('PRIVATE',str(public_messages(r)));self.assertNotIn('reference',str(public_messages(r)))
  with self.assertRaises(ValueError):adapt_row('affine_numina',{},0)
 def test_last_complete_nested_box_and_bad_tail(self):
  self.assertEqual(last_boxed_body('\\boxed{2} then \\boxed{\\frac{3}{4}} bad \\boxed{'),'\\frac{3}{4}')
  with self.assertRaises(ValueError):last_boxed_body('x'*262145)
 def test_refusals(self):
  for row in ({'question':'','final_answer':'2'},{'question':'Prove this.','final_answer':'2'},{'question':'x','final_answer':''},{'question':'x'*65537,'final_answer':'1'}):
   with self.assertRaises(ValueError):adapt_row('DeepMath-103K',row,0)
  with self.assertRaises(ValueError):adapt_row('DeepMath-103K',{'question':'x','final_answer':'1'},True)
 def test_conservative_existing_math_duplicate_key(self):
  self.assertEqual(question_key('Find $\\dfrac{1}{2}$ .'),question_key('find \\(\\frac{1}{2}\\) .'))
  self.assertNotEqual(question_key('Find 1+2.'),question_key('Find 1+3.'))
 def test_snapshot_native_contract_and_tamper_identity(self):
  r=adapt_row('DeepMath-103K',{'question':'Compute 1+1.','final_answer':'2'},0);template={'task_class':'MathTask','data':{},'task_config':{}}
  a=snapshot_row(r,template);self.assertEqual(a['data']['system_prompt'],public_messages(r)[0]['content']);self.assertEqual(a['data']['prompt'],r['question']);self.assertEqual(template['data'],{})
  self.assertNotEqual(question_key(r['question']),question_key('Compute 1+2.'))
