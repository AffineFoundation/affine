"""Disclosed MRCR shell control using only the public question/transcript.

This symbolic retrieval control is not autoregressive model sampling. It never
reads task answers or grader state; model computation must be verified separately.
"""
import re
import shlex

REVISION = 'original-mrcr-public-transcript-retrieval-v1'


def parse_public_question(question):
    if not isinstance(question, str) or len(question) > 4096:
        raise ValueError('bounded public MRCR question')
    match = re.fullmatch(
        r'Prepend ([A-Za-z0-9]{12}) to the (first|second|third|fourth|fifth|sixth|seventh|eighth) '
        r'(.+?) about (.+?) in (?:a |an )?(.+?) style\. Do not include any other text in your response\.',
        question.strip())
    if match is None:
        raise ValueError('unsupported public MRCR question')
    prefix, ordinal, kind, topic, style = match.groups()
    number = ('first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth').index(ordinal)
    requests = [f'Write {article} {kind} about {topic} in {style} style.' for article in ('a', 'an')]
    return prefix, number, requests


def answer_from_public_context(question, context):
    if not isinstance(context, str):
        raise ValueError('public transcript required')
    prefix, number, requests = parse_public_question(question)
    candidates = []
    for message in re.finditer(r'(?:^|\n)User: ([^\n]+)\n+Assistant: ?(.*?)(?=\n+User: |\Z)', context, re.S):
        if message.group(1).strip() in requests:
            candidates.append(message.group(2).strip())
    if number >= len(candidates) or not candidates[number]:
        raise ValueError('requested original response absent from public transcript')
    return prefix + candidates[number]


def shell_command(question):
    """Write the retrieved public answer in the original sandbox path.

    The command prints only a bounded acknowledgement; the original grader reads
    answer.txt. No full transcript or response is truncated into a model context.
    """
    prefix, number, requests = parse_public_question(question)
    code = ('import re\nfrom pathlib import Path\n'
            't=Path("/workspace/context.txt").read_text()\n'
            'rows=[m.group(2).strip() for m in re.finditer('
            + repr(r'(?:^|\n)User: ([^\n]+)\n+Assistant: ?(.*?)(?=\n+User: |\Z)')
            + ',t,re.S) if m.group(1).strip() in ' + repr(requests) + ']\n'
            + 'if len(rows)<=' + str(number) + ':raise ValueError("Original response absent")\n'
            + 'Path("/workspace/answer.txt").write_text(' + repr(prefix) + '+rows[' + str(number) + '])\n'
            + 'print("Public transcript answer written.")\n')
    return 'python3 -c ' + shlex.quote(code)
