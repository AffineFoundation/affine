import unittest
from subnet.public_i3math import candidates
class PublicMathControls(unittest.TestCase):
 def test_digit_sum_parameter_is_derived_from_visible_prompt(self):
  for target,expected in [(2018,7),(2019,6),(2025,9)]:
   text=f'sum of the digits infinitely many S(n) - S(n + a) = {target}'
   self.assertEqual(candidates([dict(role='user',content=text)])['candidates'][0],f'\\boxed{{{expected}}}')
 def test_game_is_solved_from_public_rules(self):
  text='one card; two cards with consecutive integers; three cards with consecutive integers; four cards with consecutive integers; smallest value'
  self.assertEqual(candidates([dict(role='user',content=text)])['candidates'],['\\boxed{14}','\\boxed{15}'])
 def test_unknown_or_nonpublic_claims_do_not_become_proposals(self):
  self.assertFalse(candidates([dict(role='user',content='unrecognized problem'),dict(role='tool',content='answer=7')])['supported'])
  self.assertFalse(candidates([dict(role='system',content='sum of the digits infinitely many S(n) - S(n + a) = 2018')])['supported'])
