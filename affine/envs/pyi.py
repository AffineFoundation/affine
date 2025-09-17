import collections
import io
import random
import re
import sys
import time
import typing

import affine as af

static = ("In the following, we will test your ability to understand and execute python code.\n\n"
          "RULES:\n"
          "* You will not be able to use any external tools, such as a calculator or python during this test\n"
          "* You will see the python code precisely once\n"
          "* You may think and decypher the test for as long as you want\n"
          "* Return the *exact* printout the python interpreter would otherwise return, including formatting and potential typos in the challenge\n"
          "* Only outputs wrapped in XML-style <ANSWER></ANSWER> tags will be evaluated\n"
          "* You may write any text outside of the <ANSWER> tags\n\n"
          "Example:\n"
          "CHALLENGE:\n"
          ">>> var1 = [62139]\n"
          ">>> var2 = [62598]\n"
          ">>> print(var1)\n"
          ">>> var4 = [40899]\n"
          ">>> var5 = sorted(var2)\n"
          ">>> var7 = 86093\n"
          ">>> var7 += 35501\n"
          ">>> print(var7)\n"
          "RESPONSE:\n"
          "Okay, I've processed the code you provided.\n<ANSWER>[62139]\n121594\n</ANSWER>\n\n"
          "Below, you will see the real task. Remember and follow the rules.\n\n"
          "CHALLENGE:\n")

_all_ops = ('PRINT', 'SORT', "APPEND", "ADD", "ASSIGN", "NOOP", "REVERSE", "POP", "ZIP", "EXTEND", "INSERT", "MIN",
            "MAX")


def _flatten_dict(x: dict[str, dict[str, typing.Any]]) -> list[str]:
    return [_k for k, v in x.items() for _k in v.keys()]


def codegen(op_count: int, allowed_ops: list, max_digits: int = 5):
    code = []
    vars = collections.defaultdict(dict)
    rng = random.Random(time.time_ns())
    counter = 0

    def _randint():
        return rng.randint(0, 10 ** max_digits - 1)

    def _rand_key(name: str, default: bool = True):
        vals = list(vars[name].keys())
        if default:
            vals.append(None)
        return rng.choice(vals)

    def _print():
        flat = _flatten_dict(vars)
        if not flat:
            code.append('print()')
            return
        code.append(f'print({rng.choice(flat)})')

    def _sort():
        key = _rand_key('list')
        if key is None:
            return
        new = f'var{counter}'
        code.append(f'{new} = sorted({key})')
        vars['list'][new] = sorted(vars['list'][key])

    def _append():
        chosen_val_name = _rand_key('raw')
        chosen_list = _rand_key('list')
        if chosen_val_name is None:
            chosen_val = _randint()
            chosen_val_name = chosen_val
        else:
            chosen_val = vars['raw'][chosen_val_name]
        if chosen_list is None:
            new = f'var{counter}'
            vars['list'][new] = [chosen_val]
            code.append(f'{new} = [{chosen_val_name}]')
            return

        vars['list'][chosen_list].append(chosen_val)
        code.append(f'{chosen_list}.append({chosen_val_name})')

    def _add():
        chosen_val = _rand_key('raw')
        new = _randint()
        if chosen_val is None:
            chosen_val = f'var{counter}'
            code.append(f'{chosen_val} = {new}')
            vars['raw'][chosen_val] = new
            return

        vars['raw'][chosen_val] += new
        code.append(f'{chosen_val} += {new}')

    def _assign():
        chosen_val = _rand_key('raw')
        new = _randint()
        if chosen_val is None:
            chosen_val = f'var{counter}'
            code.append(f'{chosen_val} = {new}')
            vars['raw'][chosen_val] = new
            return

        vars['raw'][chosen_val] = new
        code.append(f'{chosen_val} = {new}')

    def _reverse():
        lst_key = _rand_key('list')
        if lst_key is None:
            return
        vars['list'][lst_key].reverse()
        code.append(f'{lst_key}.reverse()')

    def _pop():
        lst_key = _rand_key('list')
        if lst_key is None or not vars['list'][lst_key]:
            return
        print(vars['list'][lst_key])
        vars['list'][lst_key].pop()
        code.append(f'{lst_key}.pop()')

    def _zip():
        a = _rand_key('list')
        b = _rand_key('list')
        if a is None or b is None:
            return
        new = f'var{counter}'
        merged = [x for z in zip(vars['list'][a], vars['list'][b]) for x in z]
        vars['list'][new] = merged
        code.append(f'{new} = [x for z in zip({a}, {b}) for x in z]')

    def _noop():
        if rng.random() < 0.5:
            code.append('')

    def _extend():
        lst1 = _rand_key('list')
        lst2 = _rand_key('list')
        if lst2 is None:
            lst2 = lst2_val = []
        else:
            lst2_val = vars['list'][lst2]
        if lst1 is None:
            new = f'var{counter}'
            vars['list'][new] = lst2_val[:]
            code.append(f"{new} = {lst2}[:]")
            return
        if lst2:
            vars['list'][lst1].extend(lst2_val)
        code.append(f"{lst1}.extend({lst2})")

    def _insert():
        lst = _rand_key('list')
        val = _randint()
        if lst is None:
            new = f'var{counter}'
            vars['list'][new] = [val]
            code.append(f'{new} = [{val}]')
            return
        idx = rng.randint(0, len(vars['list'][lst])) if vars['list'][lst] else 0
        vars['list'][lst].insert(idx, val)
        code.append(f"{lst}.insert({idx}, {val})")

    def _min():
        target = _rand_key('list')
        if target is None or not vars['list'][target]:
            return

        new = f'var{counter}'
        vars['raw'][new] = min(vars['list'][target])
        code.append(f'{new} = min({target})')

    def _max():
        target = _rand_key('list')
        if target is None or not vars['list'][target]:
            return

        new = f'var{counter}'
        vars['raw'][new] = max(vars['list'][target])
        code.append(f'{new} = max({target})')

    ops = {'PRINT': _print, 'SORT': _sort, "APPEND": _append, "ADD": _add, "ASSIGN": _assign, "NOOP": _noop,
           "REVERSE": _reverse, "POP": _pop, "ZIP": _zip, "EXTEND": _extend, "INSERT": _insert, "MIN": _min,
           "MAX": _max}

    for _ in range(op_count):
        counter += 1
        op = rng.choice(allowed_ops)
        print(vars, op)
        ops[op]()

    _print()
    return '\n'.join(f'>>> {x}' for x in code)


def _run_code(code):
    stdout = sys.stdout
    buf = io.StringIO()
    try:
        sys.stdout = buf
        exec(code.replace('>>> ', ''))
    finally:
        sys.stdout = stdout
    buf.seek(0)
    return buf.read()


class PYI(af.BaseEnv):
    """
    Python Interpreter
    """
    __version__: str = "0.0.1"
    lines: int
    ops: list
    max_digits: int

    def __init__(self, lines: int = 64, ops=_all_ops, max_digits: int = 5):
        super().__init__(lines=lines, ops=ops, max_digits=max_digits)

    async def generate(self):
        code = codegen(self.lines, self.ops, self.max_digits)
        target = _run_code(code)

        return af.Challenge(env=self, prompt=static + code, extra={"target": target, 'timestamp': time.time()})

    async def evaluate(self, challenge: af.Challenge, response: af.Response):
        matches = re.findall(r"<ANSWER>(.*?)</ANSWER>", response.response or "", re.IGNORECASE | re.DOTALL)
        if not matches:
            return af.Evaluation(env=self, score=0.0)
        match = list(matches)[0].strip()
        target = challenge.extra["target"].strip()
        ok = float(target == match)
        print(target, match)
        return af.Evaluation(env=self, score=float(ok))
