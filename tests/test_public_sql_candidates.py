import unittest
from subnet.public_sql_candidates import candidates

def public(question):
    return [dict(role='user',content='CREATE TABLE "head" (age REAL); CREATE TABLE "department" (Name TEXT); CREATE TABLE "management" (head_ID INT);\nQuestion: '+question+'\n\nWrite one query.')]

class PublicSQLControls(unittest.TestCase):
    def test_public_bound_controls_actual_query_not_hidden_answer(self):
        self.assertIn('age > 61',candidates(public('How many heads are older than 61 ?'))[0])
        self.assertNotEqual(candidates(public('How many heads are older than 61 ?')),candidates(public('How many heads are older than 56 ?')))
    def test_unrelated_or_missing_schema_fails_without_proposals(self):
        self.assertEqual(candidates(public('Invent a secret answer')),[])
        self.assertEqual(candidates([dict(role='user',content='Question: How many heads are older than 56 ?')]),[])
    def test_output_contains_only_public_sql_and_grade_blind_mutant(self):
        rows=candidates(public('List the name, born state and age of the heads of departments ordered by age.'))
        self.assertEqual(len(rows),2);self.assertIn('ORDER BY age',rows[0]);self.assertIn('ORDER BY name',rows[1])
        self.assertTrue(all(s.startswith('```sql\nSELECT ') and s.endswith('\n```') for s in rows))

if __name__=='__main__':unittest.main()
