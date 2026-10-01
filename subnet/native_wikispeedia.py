"""Original Wikispeedia tool execution and replay; no model provenance claim."""
from collections import deque
import hashlib
import json
import tarfile
from pathlib import Path

from .backend_jobs import canonical
from .environments import create_session


def public_candidate_harness(source, target, tools, links, max_turns=30):
    """Curated model choices from public graph data, using the common tool dialect.

    This supplies choices for model scoring; it does not assert that the model
    selected them or that a rollout has passed proof verification.
    """
    from .harness import normalize
    if type(max_turns) is not int or not 2 <= max_turns <= 30:
        raise ValueError('bounded complete navigation budget')
    clicks=[tool.get('function',{}) for tool in tools
            if isinstance(tool,dict) and
            str(tool.get('function',{}).get('name','')).endswith('click_link')]
    if len(clicks)!=1:
        raise ValueError('unambiguous original click tool schema required')
    click=clicks[0]
    parameters=click.get('parameters',{})
    if ('article' not in parameters.get('properties',{}) or
            parameters.get('properties',{}).get('article',{}).get('type')!='string'):
        raise ValueError('original public article argument required')
    route=public_path(source,target,links,max_turns-1)
    if not route:
        raise ValueError('nontrivial original navigation task required')
    unavailable='__affine_invalid_article__'
    if unavailable in links or any(unavailable in edges for edges in links.values()):
        raise ValueError('negative public article must be unavailable')
    def call(article):
        return json.dumps({'tool_call':{'name':click['name'],
                          'arguments':{'article':article}}},separators=(',',':'))
    choices=[[call(article),call(unavailable)] for article in route]
    overrides={str(i):{'candidates':candidates} for i,candidates in enumerate(choices[1:],1)}
    overrides[str(len(route))]={'candidates':['Done.','Finished.']}
    harness=normalize(dict(version='text-tools-v1',policy='candidates',
                           max_output_tokens=256,temperature=4.,top_p=1.,
                           candidates=choices[0],turn_overrides=overrides))
    return dict(route=route,click_tool=click['name'],harness=harness,
                max_turns=len(route)+1,public_starter_control_not_autonomous_search=True)


def public_path(source, target, links, max_hops=30):
    """Find a route using the public article graph, not a task reward answer."""
    queue = deque([(source, [])])
    seen = {source}
    while queue:
        article, path = queue.popleft()
        if article == target:
            return path
        if len(path) >= max_hops:
            continue
        for next_article in sorted(links.get(article, [])):
            if next_article not in seen:
                seen.add(next_article)
                queue.append((next_article, path + [next_article]))
    raise ValueError('public target unreachable within click budget')


def execute(spec, index, seed, actions):
    if spec.id != 'affine_wikispeedia':
        raise ValueError('original Wikispeedia specification required')
    if not isinstance(actions, list) or not 1 <= len(actions) <= spec.max_turns:
        raise ValueError('bounded original tool trajectory required')
    session = create_session(spec)
    try:
        reset = session.reset(index, seed)
        turns = []
        for action in actions:
            result = session.step(action)
            turns.append(dict(action=action, result=result))
        if not turns[-1]['result']['done']:
            raise ValueError('complete native trajectory required')
        return dict(schema=1, environment_id=spec.id, source_hash=spec.source_hash,
                    environment_definition_sha256=hashlib.sha256(canonical(spec.to_dict())).hexdigest(),
                    index=index, seed=seed, task_hash=reset['task_hash'],
                    reset=reset, turns=turns, reward=turns[-1]['result']['reward'])
    finally:
        session.close()


def replay(spec, artifact):
    if (artifact.get('environment_id') != spec.id or
            artifact.get('source_hash') != spec.source_hash or
            artifact.get('environment_definition_sha256') != hashlib.sha256(canonical(spec.to_dict())).hexdigest()):
        raise ValueError('approved native source binding')
    regenerated = execute(spec, artifact['index'], artifact['seed'],
                          [row['action'] for row in artifact['turns']])
    if canonical(regenerated) != canonical(artifact):
        raise ValueError('original native trajectory or outcome mismatch')
    return regenerated


def verify_public_resources(cache):
    """Check the actual graph/article files against the two original SNAP archives."""
    cache=Path(cache).resolve()
    files={};archives={}
    for name in ('wikispeedia_paths-and-graph.tar.gz','wikispeedia_articles_plaintext.tar.gz'):
        archive=cache/name
        archives[name]=dict(size=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
        with tarfile.open(archive) as tar:
            for member in tar:
                if member.isdir():
                    continue
                if not member.isfile() or member.name in files:
                    raise ValueError('regular unique original resource members required')
                raw_path=cache/member.name
                path=raw_path.resolve()
                if not path.is_relative_to(cache) or raw_path.is_symlink() or not path.is_file():
                    raise ValueError('bounded original resource member')
                original=hashlib.sha256();actual=hashlib.sha256()
                with tar.extractfile(member) as stream:
                    for chunk in iter(lambda:stream.read(1024*1024),b''):
                        original.update(chunk)
                with path.open('rb') as stream:
                    for chunk in iter(lambda:stream.read(1024*1024),b''):
                        actual.update(chunk)
                if path.stat().st_size!=member.size or original.digest()!=actual.digest():
                    raise ValueError('original SNAP extracted resource changed')
                files[member.name]=dict(size=member.size,sha256=actual.hexdigest())
    return dict(archives=archives,extracted_files=files,
                extracted_inventory_sha256=hashlib.sha256(canonical(files)).hexdigest())
