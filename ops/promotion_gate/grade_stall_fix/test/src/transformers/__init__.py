class _Tok:
    all_special_ids = ()
    def apply_chat_template(self, msgs, **kw): return [1, 2, 3]
    def decode(self, ids, skip_special_tokens=False): return ''.join('x' for _ in ids)
class AutoTokenizer:
    @staticmethod
    def from_pretrained(p): return _Tok()
