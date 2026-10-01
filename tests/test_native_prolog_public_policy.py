import unittest
from subnet.native_prolog_public_policy import candidates
class Tests(unittest.TestCase):
 def test_shared_public_program_single_constraint_mutation(self):
  public={'kind':'nqueens','starter_file':'board_size(12).\nsolve(_) :- fail.  %% TODO: replace this\n'}
  positive,negative=candidates(public)
  self.assertEqual(positive.replace('abs(Q-R) #\\= D','abs(Q-R) #\\=0'),negative)
  self.assertEqual(len(positive),len(negative)+1)
  for command in (positive,negative):self.assertIn('board_size(12).',command);self.assertIn('all_distinct(Qs)',command)
 def test_unsupported_public_problem_rejected(self):
  with self.assertRaises(ValueError):candidates({'kind':'sudoku'})
if __name__=='__main__':unittest.main()
