"""Prospective public-only arithmetic controls for original RCore prompts."""
import ast,math,re
from fractions import Fraction


def arithmetic_candidates(messages):
    if not isinstance(messages,list):raise ValueError('public message list')
    text='\n'.join(m.get('content','') for m in messages if m.get('role')=='user')
    if len(text)>4096:raise ValueError('bounded public prompt')
    match=re.fullmatch(r'Evaluate (.+)\.\s*The answer is a number\.',text.strip())
    if not match:raise ValueError('unsupported public RCore arithmetic prompt')
    tree=ast.parse(match[1],mode='eval')
    if len(list(ast.walk(tree)))>96:raise ValueError('bounded arithmetic syntax')
    def calculate(node):
        if isinstance(node,ast.Constant) and type(node.value)in(int,float):
            literal=ast.get_source_segment(match[1],node)
            exponent=re.search(r'[eE]([+-]?\d+)',literal)
            if len(literal)>64 or (exponent and abs(int(exponent[1]))>128):raise ValueError('bounded numeric literal')
            value=Fraction(literal)
        elif isinstance(node,ast.UnaryOp)and isinstance(node.op,(ast.UAdd,ast.USub)):
            value=calculate(node.operand)*(1 if isinstance(node.op,ast.UAdd)else -1)
        elif isinstance(node,ast.BinOp)and isinstance(node.op,(ast.Add,ast.Sub,ast.Mult,ast.Div,ast.Mod)):
            a,b=calculate(node.left),calculate(node.right)
            if isinstance(node.op,ast.Add):value=a+b
            elif isinstance(node.op,ast.Sub):value=a-b
            elif isinstance(node.op,ast.Mult):value=a*b
            elif isinstance(node.op,ast.Div):value=a/b
            else:value=a%b
        else:raise ValueError('unsupported public arithmetic syntax')
        if value.numerator.bit_length()>4096 or value.denominator.bit_length()>4096 or not math.isfinite(value)or abs(value)>1e12 or (value and abs(value)<Fraction(1,10**128)):raise ValueError('bounded finite arithmetic')
        return value
    value=calculate(tree.body)
    def render(number):
        return format(float(number),'.17g')
    candidates=['<answer>'+render(v)+'</answer>' for v in (value,value+1)]
    if len(set(candidates))!=2:raise ValueError('distinct public arithmetic candidates')
    return candidates
