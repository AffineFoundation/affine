import random
import unittest
from ops.population_index import _index, _matches, FIELDS

class PairIndex(unittest.TestCase):
 def assert_parity(self,rows,pairs):
  index=_index(rows)
  for pair in pairs:
   old=[r for r in rows if all(r[k]==v for k,v in pair.items())]
   new=_matches(index,rows,pair)
   self.assertEqual(old,new)
   self.assertEqual([id(x)for x in old],[id(x)for x in new])
 def test_random_and_duplicate_rows_preserve_order_multiplicity(self):
  rng=random.Random(3)
  rows=[{k:str(rng.randrange(7))for k in FIELDS}for _ in range(400)]
  rows=rows+rows[::7]
  self.assert_parity(rows,rows+[dict.fromkeys(FIELDS,'absent')])
 def test_non_string_values_use_original_equality(self):
  rows=[dict.fromkeys(FIELDS,'a')]
  for value in (None,0,True,[],{},['a']):
   for field in FIELDS:
    pair=dict(rows[0]);pair[field]=value;self.assert_parity(rows,[pair])
 def test_metadata_is_not_used(self):
  row=dict.fromkeys(FIELDS,'x');row['unrelated']=3
  self.assert_parity([row],[dict.fromkeys(FIELDS,'x')])
 def test_exact_pair_and_empty(self):
  self.assert_parity([], [dict.fromkeys(FIELDS,'x')])
  self.assert_parity([dict.fromkeys(FIELDS,'x')],[])
if __name__=='__main__':unittest.main()
