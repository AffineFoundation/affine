"""Question-only adapters for pinned prose-math corpora using the original MATH grader.

These corpora are separate from Affine's Lean Numina environment. Counts produced
here establish extractable, bounded, deduplicated rows; full native/model audit
remains an explicit later qualification gate.
"""
from __future__ import annotations
import copy,hashlib,json,re,unicodedata
from pathlib import Path

VERSION='pinned-question-only-math-corpora-v1'
SYSTEM='Solve the math problem. Reason step by step in plain text, then end your response with the final answer in `\\boxed{}`. Emit exactly one `\\boxed{}` block and nothing after it.'
MAX_QUESTION_CHARS=65536
MAX_REFERENCE_CHARS=4096
MAX_SOLUTION_CHARS=262144


def question_key(question):
    if not isinstance(question,str) or not question.strip():raise ValueError('empty question')
    normalized=unicodedata.normalize('NFKC',question)
    normalized=normalized.replace('\\dfrac','\\frac').replace('\\tfrac','\\frac')
    for marker in ('\\left','\\right','\\[','\\]','\\(','\\)','$'):
        normalized=normalized.replace(marker,'')
    normalized=re.sub(r'\s+','',normalized).casefold()
    return hashlib.sha256(normalized.encode()).hexdigest()


def last_boxed_body(text):
    if not isinstance(text,str):return None
    if len(text)>MAX_SOLUTION_CHARS:raise ValueError('solution length bound')
    found=None;opening='\\boxed{';start=text.find(opening)
    while start!=-1:
        depth=0
        for i in range(start+len(opening)-1,len(text)):
            if text[i]=='{':depth+=1
            elif text[i]=='}':
                depth-=1
                if depth==0:
                    found=text[start+len(opening):i];break
        start=text.find(opening,start+len(opening))
    return found


def adapt_row(corpus,row,source_index):
    if type(source_index) is not int or source_index<0:raise ValueError('source index')
    if corpus=='DeepMath-103K':
        question=row.get('question');answer=row.get('final_answer')
        if isinstance(answer,str) and answer.strip().startswith('\\boxed{'):
            answer=last_boxed_body(answer)
        category=str(row.get('topic') or '');difficulty=str(row.get('difficulty') or '')
    elif corpus=='NuminaMath-CoT':
        question=row.get('problem');answer=last_boxed_body(row.get('solution'))
        category=str(row.get('source') or '');difficulty=''
    else:raise ValueError('unapproved corpus')
    if not isinstance(question,str) or not question.strip():raise ValueError('empty question')
    if len(question)>MAX_QUESTION_CHARS:raise ValueError('question length bound')
    if not isinstance(answer,str) or not answer.strip():raise ValueError('missing final answer')
    answer=answer.strip()
    if len(answer)>MAX_REFERENCE_CHARS:raise ValueError('reference length bound')
    # Proof-only prompts are outside a boxed numeric/symbolic final-answer pilot.
    if re.match(r'^\s*(prove|show|demonstrate)\b',question,re.I):raise ValueError('proof-only prompt')
    return {'corpus':corpus,'source_index':source_index,'question':question,
            'reference':answer,'category':category,'difficulty':difficulty,
            'question_key':question_key(question)}


def public_messages(row):
    # Explicitly omit references/solutions; preserve the original question bytes.
    return [{'role':'system','content':SYSTEM},{'role':'user','content':row['question']}]


def snapshot_row(row,template):
    out=copy.deepcopy(template);data=out['data']
    assert out['task_class']=='MathTask'
    data.update(idx=row['source_index'],name='corpus-'+row['corpus'].lower()+'-'+hashlib.sha256(row['question'].encode()).hexdigest()[:20],
                problem=row['question'],prompt=row['question'],answer=row['reference'],
                subject=row['category'],level=row['difficulty'],system_prompt=SYSTEM)
    return out


def write_snapshot(rows,path,template):
    path=Path(path);payload=[snapshot_row(row,template) for row in rows]
    body=json.dumps(payload,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
    path.write_bytes(body)
    return {'size':len(body),'sha256':hashlib.sha256(body).hexdigest(),'rows':len(rows)}
