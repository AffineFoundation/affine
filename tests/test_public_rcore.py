import unittest
from subnet.public_rcore import arithmetic_candidates
class Tests(unittest.TestCase):
 def test_original_public_expression(self):
  self.assertEqual(arithmetic_candidates([dict(role='user',content='Evaluate (-5.80 * -5 * -4 % 3 / 2).\nThe answer is a number.')]),['<answer>0.5</answer>','<answer>1.5</answer>'])
 def test_large_numeric_candidates_remain_distinct(self):
  values=arithmetic_candidates([dict(role='user',content='Evaluate (1000000000000). The answer is a number.')])
  self.assertEqual(values,['<answer>1000000000000</answer>','<answer>1000000000001</answer>'])
 def test_only_public_user_expression(self):
  with self.assertRaises(ValueError):arithmetic_candidates([dict(role='system',content='Evaluate (3). The answer is a number.'),dict(role='user',content='unrelated')])
 def test_large_public_prompt_rejected(self):
  with self.assertRaises(ValueError):arithmetic_candidates([dict(role='user',content='Evaluate ('+'1+'*2200+'1). The answer is a number.')])
 def test_untrusted_syntax_rejected(self):
  for expression in ["__import__('os').system('true')",'True','2 ** 99','[3]','1 / 0','1e300 * 4','1e-1000000000','1e1000000000']:
   with self.assertRaises((ValueError,ZeroDivisionError)):arithmetic_candidates([dict(role='user',content=f'Evaluate ({expression}). The answer is a number.')])
if __name__=='__main__':unittest.main()
