import re, ast
S = ast.literal_eval(re.search(r"^_ALLAN_SRC = (.*)$", open('stage/probe_worker.py').read(), re.M).group(1))
ns = {}; exec('import re\nfrom math_verify import parse, verify\n' + S, ns)
def ok(gold, text, prompt=''):
    g, v, p = ns['lenient_project'](text); return bool(g is not None and ns['lenient_equal'](gold, g, prompt)[0])
T = '<think>x</think>\n'
cases = [  # (gold, reply, prompt, expected)
 ('25200', T + '$$20 \\times 30 \\times 42 = 25,\\!200$$\n\\boxed{25200}', '', True),
 ('25200', T + '\\boxed{25201}', '', False),
 ('819', T + 'The answer is $\\boxed{819}$.', '', True),
 ('819', T + 'The answer is $\\boxed{818}$.', '', False),
 ('\\text{A, D, F, G, H}', T + '\\boxed{A, D, F, G, H}', '', True),
 ('\\text{A, D, F, G, H}', T + '\\boxed{A, D, F, G}', '', False),
 ('119', T + '\\boxed{C}', '$\\textbf{(A)}\\ 100 \\qquad \\textbf{(B)}\\ 110 \\qquad \\textbf{(C)}\\ 119 \\qquad \\textbf{(D)}\\ 120$', True),
 ('119', T + '\\boxed{B}', '$\\textbf{(A)}\\ 100 \\qquad \\textbf{(B)}\\ 110 \\qquad \\textbf{(C)}\\ 119 \\qquad \\textbf{(D)}\\ 120$', False),
 ('\\begin{pmatrix} 6 \\\\ -17 \\end{pmatrix}', T + '\\boxed{(6, -17)}', '', True),
 ('\\begin{pmatrix} 6 \\\\ -17 \\end{pmatrix}', T + '\\boxed{(6, 17)}', '', False),
 ('(-\\infty,-\\frac{1}{5})', T + '\\boxed{k < -\\frac{1}{5}}', '', True),
 ('(-\\infty,-\\frac{1}{5})', T + '\\boxed{k > -\\frac{1}{5}}', '', False),
 ('\\begin{pmatrix} -5 & 0 \\\\ 0 & -5 \\end{pmatrix}', T + '\\boxed{-5\\mathbf{I}}', '', True),
 ('\\begin{pmatrix} -5 & 0 \\\\ 0 & -5 \\end{pmatrix}', T + '\\boxed{5\\mathbf{I}}', '', False),
 ('-32 + 32i\\sqrt{3}', T + '\\boxed{-32 + 32\\sqrt{3}i}', '', True),
 ('-32 + 32i\\sqrt{3}', T + '\\boxed{-32 - 32\\sqrt{3}i}', '', False),
 ('5', T + 'no box here, answer 5', '', False),
 ('5', T + '\\boxed{}', '', False),
 ('5', T + '\\boxed{4} wait \\boxed{6}', '', False),
]
bad = [(g, r, e) for g, r, p, e in cases if ok(g, r, p) != e]
print('%d/%d PASS' % (len(cases) - len(bad), len(cases))); [print('FAIL', b) for b in bad]
