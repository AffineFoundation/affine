"""Public counters for committed inputs; signatures do not certify sampled traces."""
import base64,hashlib,json
from nacl.signing import VerifyKey
AUTHORITY='3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd'
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def authenticated(document,signer):
 if not isinstance(document,dict)or set(document)!={'payload','signature','signer'}or document['signer']!=signer:raise ValueError('expected signed document')
 VerifyKey(bytes.fromhex(signer)).verify(canonical(document['payload']),base64.b64decode(document['signature'],validate=True));return document['payload']
def project(document,manifest,identity_uids,authority=AUTHORITY):
 try:
  if document.get('version')!='committed-unaudited-training-v1':raise ValueError('learner population version')
  population=document['population'];epoch=manifest['epoch'];checkpoint=manifest['checkpoint']['id'];source=manifest['source_bundle']['sha256']
  if population['assurance']!='unaudited'or population['epoch']!=epoch or population['checkpoint']!=checkpoint:raise ValueError('learner scope')
  committed={};grid=[0]*256;eligible_grid=[0]*256;outside=0
  for row in population['committed_inventory']:
   miner=row['miner'];signed=row['commitment_document'];p=authenticated(signed,miner)
   if p['epoch']!=epoch or p['checkpoint']!=checkpoint or p['source']!=source or p['miner']!=miner or digest(signed)!=row['commitment_sha256']or p['version']!='small-commitment-pairs-v2':raise ValueError('committed source/epoch/hash')
   for child in p['batches']:
    key=(miner,child['slot'])
    if key in committed:raise ValueError('duplicate declared slot')
    committed[key]=(row,child);uid=identity_uids.get(miner)
    if type(uid)is int and 0<=uid<256:grid[uid]+=1
    else:outside+=1
  eligible=set();inventory=[]
  for row in document['submissions']:
   signed=row['learner_admission'];p=authenticated(signed,authority);key=(p['miner_identity'],p['slot'])
   if key in eligible or key not in committed or p['version']!='committed-unaudited-training-v1'or p['assurance']!='unaudited'or p['epoch']!=epoch or p['checkpoint']!=checkpoint or p['source_sha256']!=source:raise ValueError('eligible admission scope')
   original,child=committed[key]
   if p['commitment_sha256']!=original['commitment_sha256']or digest(p['original_commitment'])!=original['commitment_sha256']or p['batch_sha256']!=child['batch_sha256']or p['proof_sha256']!=child['sha256']or p['document_sha256']!=child['training_sha256']or row['sha256']!=p['document_sha256']or row['size']!=p['document_size']or p['document_size']!=child['training_size']:raise ValueError('exact declared child admission')
   inventory.append({'learner_admission_sha256':digest(signed),'sha256':row['sha256'],'size':row['size']});eligible.add(key);uid=identity_uids.get(key[0])
   if type(uid)is int and 0<=uid<256:eligible_grid[uid]+=1
  if population['committed_count']!=len(committed)or population['eligible_count']!=len(eligible)or sorted(population['eligible_inventory'],key=lambda r:r['learner_admission_sha256'])!=sorted(inventory,key=lambda r:r['learner_admission_sha256']):raise ValueError('actual inventory counts')
  return dict(submitted=len(committed),learner_eligible=len(eligible),learner_excluded=len(committed)-len(eligible),submitted_grid=grid,eligible_grid=eligible_grid,unassigned=outside,submitting_identities=len({key[0]for key in committed}),eligible_identities=len({key[0]for key in eligible}),input_assurance='unaudited',proof_verification_claimed=False,source='authenticated-committed-learner-population',provenance_sha256=digest(document))
 except Exception:
  # Public state never contains the submitted documents, capabilities, or traces.
  # Invalid projection is unavailable; never silently invent a zero population.
  return None
