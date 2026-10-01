"""Image entrypoint: original GeneralAgent DB/tools and separate native grader.

The operator builds distinct actor/grader images. This file never downloads
code, accepts an uploaded implementation, or exposes a private grader to the
actor. MCP argument validation and ToolError observations use pinned MCP.
"""
import ast
import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
from types import SimpleNamespace

def main():
    root=Path('/fixture');manifest=json.loads((root/'manifest.json').read_text())
    actual={p.name for p in root.iterdir() if p.is_file()}
    if actual!=set(manifest['files'])|{'manifest.json'}:raise ValueError('native fixture file closure')
    for name,digest in manifest['files'].items():
        if Path(name).name!=name or hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('native fixture source changed')
    package=types.ModuleType('general_agent');package.__path__=[];sys.modules['general_agent']=package
    def load(name,path):
        spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
    load('general_agent.tools',root/'base.py');module=load('fixture_tools',root/'tools.py')
    if sys.argv[1]=='grader':
        # Original method bodies and normal TaskDB serialization are retained;
        # only decorators/framework scheduling are outside this controlled
        # fixture boundary. It is not the full verifiers orchestrator.
        tree=ast.parse((root/'original-taskset.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GeneralAgentTask')
        functions=[n for n in cls.body if isinstance(n,ast.AsyncFunctionDef) and n.name in ('checks','solved')]
        if len(functions)!=2:raise ValueError('native original grader methods')
        for function in functions:function.decorator_list=[]
        program=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),ast.ClassDef(name='OriginalGrader',bases=[],keywords=[],body=functions,decorator_list=[])],type_ignores=[]);ast.fix_missing_locations(program)
        namespace=dict(Path=Path,json=json,load_task_attrs=lambda path,*attrs:tuple(getattr(module,n,None) for n in attrs))
        exec(compile(program,'approved-original-native-grader','exec'),namespace)
        state=json.load(sys.stdin);grader=namespace['OriginalGrader']();grader.data=SimpleNamespace(dir=root)
        trace=SimpleNamespace(state=SimpleNamespace(db=state),metrics={});trace.metrics=asyncio.run(grader.checks(trace));reward=asyncio.run(grader.solved(trace))
        print(json.dumps(dict(metrics=trace.metrics,reward=reward,original_source_sha256=manifest['files']['original-taskset.py'])),flush=True);return
    if sys.argv[1]!='actor':raise ValueError('unapproved native role')
    from mcp.server.fastmcp import FastMCP
    from mcp.server.fastmcp.exceptions import ToolError
    tools=module.TaskTools(module.TaskDB.load(root/'db.json'));server=FastMCP('approved-original-general-agent')
    for name,method in tools.tool_methods.items():server.tool(name=name)(method)
    async def call(name,arguments):
        try:return await server._tool_manager.call_tool(name,arguments)
        except ToolError as error:return str(error)
    for line in sys.stdin:
        request=json.loads(line);operation=request['operation']
        if operation=='list_tools':
            result=[dict(type='function',function=dict(name=t.name,description=t.description or '',parameters=t.inputSchema)) for t in asyncio.run(server.list_tools())]
        elif operation=='call':result=asyncio.run(call(request['name'],request['arguments']))
        elif operation=='state':result=tools.db.model_dump(mode='json')
        else:raise ValueError('unapproved native operation')
        print(json.dumps(dict(result=result,state_hash=tools.db.get_hash())),flush=True)

if __name__=='__main__':main()
