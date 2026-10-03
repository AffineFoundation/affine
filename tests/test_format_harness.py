import copy
import unittest

from subnet.harness import HARNESS_REGISTRY, normalize, render


class RecordingTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        self.messages = copy.deepcopy(messages)
        return [11, 12]


class FormatHarnessTests(unittest.TestCase):
    def config(self):
        return dict(version='text-tools-format-long-v1', policy='autoregressive',
                    max_output_tokens=1024, temperature=.8, top_p=1.,
                    response_format_instruction='Finish with <answer>...</answer>.')

    def test_render_preserves_native_messages_and_conversation(self):
        messages = [{'role': 'system', 'content': 'Native instructions'},
                    {'role': 'user', 'content': 'First question'},
                    {'role': 'assistant', 'content': 'Previous answer'},
                    {'role': 'user', 'content': 'Next question'}]
        original = copy.deepcopy(messages)
        tokenizer = RecordingTokenizer()
        self.assertEqual(render(tokenizer, messages, config=self.config()), [11, 12])
        self.assertEqual(messages, original)
        self.assertEqual(tokenizer.messages[:-1], original[:-1])
        self.assertEqual(tokenizer.messages[-1]['content'],
                         'Next question\n\nResponse format: ' + self.config()['response_format_instruction'])

    def test_instruction_cannot_silently_use_legacy_version(self):
        for version in ('text-tools-v1', 'text-tools-long-v2', 'text-tools-window-v1'):
            with self.subTest(version=version), self.assertRaises(ValueError):
                normalize({**self.config(), 'version': version, 'max_output_tokens': 256})

    def test_missing_or_unbounded_instruction_refused(self):
        for value in (None, '', '   ', True, ['text'], 'x' * 513):
            with self.subTest(value=type(value).__name__), self.assertRaises(ValueError):
                normalize({**self.config(), 'response_format_instruction': value})

    def test_missing_or_nontext_public_user_message_refused(self):
        for messages in ([], [{'role': 'system', 'content': 'Only system'}],
                         [{'role': 'user', 'content': [{'type': 'text', 'text': 'Q'}]}]):
            with self.assertRaises(ValueError):
                render(RecordingTokenizer(), messages, config=self.config())

    def test_legacy_render_and_sampling_boundary_preserved(self):
        tokenizer = RecordingTokenizer()
        messages = [{'role': 'user', 'content': 'Original question'}]
        legacy = {k: v for k, v in self.config().items() if k != 'response_format_instruction'}
        legacy['version'] = 'text-tools-long-v2'
        render(tokenizer, messages, config=legacy)
        self.assertEqual(tokenizer.messages, messages)
        for key in ('sample', 'action', 'observations'):
            self.assertIs(HARNESS_REGISTRY[legacy['version']][key],
                          HARNESS_REGISTRY[self.config()['version']][key])
        for key in ('policy', 'max_output_tokens', 'temperature', 'top_p'):
            self.assertEqual(normalize(legacy)[key], normalize(self.config())[key])


if __name__ == '__main__':
    unittest.main()
