import shlex,tempfile,subprocess,unittest
from pathlib import Path
from ops.probe_numina_native_tactics import tactic_command
class PublicStarterTests(unittest.TestCase):
 def test_public_statement_preserved(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'proof.lean';prefix='import Mathlib\ntheorem test : 1 + 1 = 2 := by\n  '
   p.write_text(prefix+'sorry\n');subprocess.run(shlex.split(tactic_command(str(p))),check=True)
   self.assertTrue(p.read_text().startswith(prefix));self.assertNotIn('sorry',p.read_text());self.assertIn('aesop',p.read_text())
 def test_path_quoted_no_shell_side_effect(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'$(oops) proof.lean';p.write_text('theorem test : True := by\n sorry\n')
   subprocess.run(shlex.split(tactic_command(str(p))),check=True);self.assertIn('first',p.read_text())
 def test_absent_placeholder_fails(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'proof.lean';p.write_text('theorem test : True := by trivial\n')
   result=subprocess.run(shlex.split(tactic_command(str(p))),capture_output=True);self.assertNotEqual(result.returncode,0)
if __name__=='__main__':unittest.main()
