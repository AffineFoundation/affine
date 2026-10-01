"""Bounded Pydantic proposals derived exclusively from the visible Python schema.

This parses AST; it never executes schema code or reads task/grader fields.
Unsupported annotations and constraints fail explicitly. Native grading is a
separate control and these proposals alone claim no valid model rollout.
"""
import ast
import json
import re

REVISION = 'public-pydantic-ast-proposals-v1'


def _name(node):
    return node.id if isinstance(node,ast.Name) else node.attr if isinstance(node,ast.Attribute) else ''


def _literal(node):
    try: return ast.literal_eval(node)
    except (ValueError,TypeError): raise ValueError('unsupported public schema literal') from None


def _field(node, variable):
    return node.attr if isinstance(node,ast.Attribute) and isinstance(node.value,ast.Name) and node.value.id==variable else None


def _bounds(class_node):
    result = {}
    for method in class_node.body:
        if not isinstance(method,(ast.FunctionDef,ast.AsyncFunctionDef)): continue
        variables = {a.arg for a in method.args.args}
        for node in ast.walk(method):
            if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.Not):
                for variable in variables:
                    field = _field(node.operand,variable)
                    if field: result.setdefault(field,{})['nonempty']=True
            if isinstance(node,ast.Compare):
                # Public predicates of the form low <= value <= high.
                if len(node.ops)==2 and isinstance(node.left,ast.Constant) and isinstance(node.comparators[1],ast.Constant):
                    for variable in variables:
                        field = _field(node.comparators[0],variable)
                        if field and isinstance(node.ops[0],(ast.Lt,ast.LtE)) and isinstance(node.ops[1],(ast.Lt,ast.LtE)):
                            result.setdefault(field,{}).update(ge=node.left.value,le=node.comparators[1].value)
    return result


def proposals(messages):
    visible = '\n'.join(m['content'] for m in messages if m.get('role') in {'system','user'} and isinstance(m.get('content'),str))
    if len(visible)>100000: raise ValueError('public schema text budget')
    blocks = re.findall(r'```python\s*([\s\S]*?)```',visible)
    names = re.findall(r'model named\s+([A-Za-z_][A-Za-z_0-9]*)',visible)
    if len(blocks)!=1 or len(names)!=1: raise ValueError('one public schema and model name required')
    try: tree = ast.parse(blocks[0])
    except (SyntaxError,RecursionError): raise ValueError('bounded valid public Python schema required') from None
    if sum(1 for _ in ast.walk(tree))>10000: raise ValueError('public schema AST budget')
    classes={n.name:n for n in tree.body if isinstance(n,ast.ClassDef)}
    if names[0] not in classes: raise ValueError('public root schema missing')
    required = {}

    def value(annotation, constraints, label, depth):
        if depth>12: raise ValueError('public schema nesting budget')
        typ = _name(annotation)
        if isinstance(annotation,ast.Subscript):
            typ=_name(annotation.value); args=list(annotation.slice.elts) if isinstance(annotation.slice,ast.Tuple) else [annotation.slice]
            if typ in {'Optional','Union'}:
                chosen=next((a for a in args if not (isinstance(a,ast.Constant) and a.value is None)),None)
                if chosen is None: return None
                return value(chosen,constraints,label,depth+1)
            if typ=='Literal': return _literal(args[-1])
            if typ in {'List','list','Sequence','Set','set'}:
                count=max(int(constraints.get('min_length',0)),int(constraints.get('nonempty',False)))
                if count>4: raise ValueError('public collection budget')
                return [value(args[0],{},label,depth+1) for _ in range(count)]
            if typ in {'Dict','dict','Mapping'}: return {}
            if typ in {'Tuple','tuple'}:
                if any(isinstance(a,ast.Constant) and a.value is Ellipsis for a in args): raise ValueError('variable public tuple unsupported')
                return [value(a,{},label,depth+1) for a in args]
            if typ=='Annotated':
                merged=dict(constraints)
                for a in args[1:]:
                    if isinstance(a,ast.Call) and _name(a.func)=='Field': merged.update({k.arg:_literal(k.value) for k in a.keywords})
                return value(args[0],merged,label,depth+1)
            raise ValueError('unsupported public generic '+typ)
        if typ in classes:
            cls=classes[typ]
            if any(_name(base)=='Enum' for base in cls.bases):
                constants=[_literal(n.value) for n in cls.body if isinstance(n,ast.Assign)]
                if not constants: raise ValueError('public enum values missing')
                return constants[0]
            return instance(typ,depth+1)
        if typ in {'str','EmailStr','HttpUrl','AnyUrl','IPvAnyAddress'}:
            if typ=='EmailStr': return 'example@example.com'
            if typ in {'HttpUrl','AnyUrl'}: return 'https://example.com'
            if typ=='IPvAnyAddress': return '127.0.0.1'
            if constraints.get('pattern') or constraints.get('regex'): raise ValueError('public regex synthesis unsupported')
            length=max(1,int(constraints.get('min_length',1)))
            if length>128 or length>constraints.get('max_length',128): raise ValueError('public string budget')
            return 'x'*length
        if typ in {'int','float','Decimal','PositiveInt','PositiveFloat','NonNegativeInt','NonNegativeFloat'}:
            lower=constraints.get('ge',0)
            if 'gt' in constraints: lower=max(lower,constraints['gt']+1)
            if typ.startswith('Positive'): lower=max(lower,1)
            upper=constraints.get('le',lower+100)
            if 'lt' in constraints: upper=min(upper,constraints['lt']-1)
            if lower>upper: raise ValueError('unsupported public numeric interval')
            number=lower
            return int(number) if typ in {'int','PositiveInt','NonNegativeInt'} else float(number)
        if typ=='bool': return False
        if typ=='date': return '2030-01-02' if any(x in label for x in ['end','deadline','due']) else '2030-01-01'
        if typ=='time': return '10:00:00' if 'end' in label else '09:00:00'
        if typ=='datetime': return '2030-01-02T10:00:00' if 'end' in label else '2030-01-01T09:00:00'
        if typ=='UUID': return '00000000-0000-4000-8000-000000000001'
        raise ValueError('unsupported public annotation '+typ)

    def instance(name,depth):
        cls=classes[name]; result={}; mandatory=[]; inferred=_bounds(cls)
        for node in cls.body:
            if not isinstance(node,ast.AnnAssign) or not isinstance(node.target,ast.Name): continue
            label=node.target.id
            if label=='model_config': continue
            constraints=dict(inferred.get(label,{})); default=node.value; needed=default is None
            if isinstance(default,ast.Call) and _name(default.func)=='Field':
                constraints.update({k.arg:_literal(k.value) for k in default.keywords if k.arg not in {'description','title','examples','default_factory'}})
                needed=not default.args and 'default' not in constraints and not any(k.arg=='default_factory' for k in default.keywords)
                if default.args: needed=isinstance(default.args[0],ast.Constant) and default.args[0].value is Ellipsis
            if needed or constraints.get('nonempty'):
                key=constraints.get('alias',label); result[key]=value(node.annotation,constraints,label,depth+1)
                if needed: mandatory.append(key)
        required[name]=mandatory
        return result

    positive=instance(names[0],0); keys=required[names[0]]
    if not keys: raise ValueError('negative mutation needs a required public root field')
    negative=dict(positive); key=keys[0]
    changed=''.join('z' if c!='z' else 'y' for c in key)
    if changed in negative: raise ValueError('public mutation collision')
    negative[changed]=negative.pop(key)
    render=lambda v:'```json\n'+json.dumps(v,separators=(',',':'))+'\n```'
    return [render(positive),render(negative)]
