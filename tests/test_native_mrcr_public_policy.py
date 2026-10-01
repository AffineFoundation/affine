import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

from subnet.native_mrcr_public_policy import answer_from_public_context, shell_command


class PublicMRCR(unittest.TestCase):
    question = 'Prepend abcdef123456 to the second poem about stars in a humorous style. Do not include any other text in your response.'
    context = ('User: Write a poem about stars in humorous style.\n\nAssistant: first\n\n'
               'User: Write a poem about stars in technical style.\n\nAssistant: unrelated\n\n'
               'User: Write a poem about stars in humorous style.\n\nAssistant: second\n\n'
               'User: Final follow-up\n\nAssistant:')

    def test_exact_public_occurrence_and_style(self):
        self.assertEqual(answer_from_public_context(self.question, self.context), 'abcdef123456second')

    def test_no_fallback_when_original_occurrence_missing(self):
        with self.assertRaises(ValueError):
            answer_from_public_context(self.question.replace('second', 'third'), self.context)

    def test_unsupported_questions_rejected(self):
        for question in ['invent a task', self.question.replace('abcdef123456', 'short'), self.question.replace('second', 'last')]:
            with self.subTest(question=question), self.assertRaises(ValueError):
                shell_command(question)

    def test_shell_retrieval_uses_public_file_and_bounded_observation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root/'context.txt').write_text(self.context)
            args = shlex.split(shell_command(self.question))
            self.assertEqual(args[:2], ['python3', '-c'])
            code = args[2].replace('/workspace/context.txt', str(root/'context.txt')).replace('/workspace/answer.txt', str(root/'answer.txt'))
            result = subprocess.run(['python3', '-c', code], check=True, capture_output=True, text=True)
            self.assertEqual((root/'answer.txt').read_text(), 'abcdef123456second')
            self.assertEqual(result.stdout, 'Public transcript answer written.\n')


if __name__ == '__main__':
    unittest.main()
