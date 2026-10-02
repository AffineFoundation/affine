"""Snapshot-only corpus shards; aliases preserve the original MathTask grader."""
import hashlib,re,os
from pathlib import Path
VERSION='original-math-corpus-shard-v1'
SHARD_ROWS=8192
CORPORA={'deepmath103k':('DeepMath-103K','5cf055d1fe3d7a2eb19719ac020211469736ae44'),
         'numinamath_cot':('NuminaMath-CoT','9d8d210c9f6a36c8f3cd84045668c9b7800ef517')}
PATTERN=re.compile(r'math_corpus_(deepmath103k|numinamath_cot)_(train|heldout|official_test)_([0-9]{3})\Z')
def is_corpus_id(value):return isinstance(value,str) and PATTERN.fullmatch(value) is not None

def validate(spec,root=None):
 match=PATTERN.fullmatch(spec.id)
 if not match or spec.adapter!='prime_v1' or spec.version!=VERSION:raise ValueError('corpus provider version/adapter')
 slug,fold,ordinal=match.groups();binding=spec.config.get('math_corpus_asset')
 if not isinstance(binding,dict) or set(binding)!={'version','corpus','upstream_revision','catalog_sha256','fold','shard','rows','sha256','size','compressed_sha256','compressed_size','path','provider_sha256'}:raise ValueError('corpus binding shape')
 corpus,revision=CORPORA[slug]
 if binding['version']!=VERSION or binding['corpus']!=corpus or binding['upstream_revision']!=revision or binding['fold']!=fold or type(binding['shard']) is not int or binding['shard']!=int(ordinal):raise ValueError('corpus origin/fold/shard')
 for key in ('catalog_sha256','sha256','compressed_sha256','provider_sha256'):
  if not isinstance(binding[key],str) or re.fullmatch('[0-9a-f]{64}',binding[key]) is None:raise ValueError('corpus digest')
 if binding['provider_sha256']!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():raise ValueError('corpus provider source')
 if type(binding['rows']) is not int or not 1<=binding['rows']<=SHARD_ROWS or spec.num_samples!=binding['rows']:raise ValueError('corpus row bound')
 if spec.max_turns!=1 or type(spec.success_reward) not in (int,float) or spec.success_reward!=1.0:raise ValueError('original MathTask contract')
 from .math_corpus_assets import validate_asset
 validate_asset(binding)
 if spec.config.get('task_snapshot')!=binding['path']:raise ValueError('corpus snapshot binding')
 if root is not None:
  path=asset_path(binding,root)
  if path.is_symlink() or not path.is_file() or path.stat().st_size!=binding['size'] or hashlib.sha256(path.read_bytes()).hexdigest()!=binding['sha256']:raise ValueError('corpus asset not hydrated/exact')
 return binding

def taskset_source(spec):
 validate(spec)
 return ('affine_math_v1','MathTaskset')


def asset_path(binding,source_root):
 """Resolve a public immutable asset outside the readonly executable cache."""
 from .math_corpus_assets import validate_asset
 validate_asset(binding)
 external=os.environ.get('AFFINE_MATH_CORPUS_ASSET_ROOT')
 root=Path(external) if external is not None else Path(source_root)
 if not root.is_absolute() or root.absolute()!=root.resolve():raise ValueError('corpus asset root must be absolute and unaliased')
 path=root/binding['path']
 cursor=path
 while cursor!=root:
  if cursor.is_symlink():raise ValueError('corpus asset symlink')
  cursor=cursor.parent
 return path
