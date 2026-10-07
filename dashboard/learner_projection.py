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
   # A signed declaration can precede the small training-document upload.
   # Count the captured inventory, not missing/deferred declared children.
   captured=None
   if 'training_documents'in row:
    children={c['slot']:c for c in p['batches']}
    if len(children)!=len(p['batches']):raise ValueError('duplicate declared capture slot')
    captured={}
    for capture in row['training_documents']:
     slot=capture['slot'];child=children.get(slot)
     if type(slot)is not int or slot in captured or child is None or capture['sha256']!=child['training_sha256']or type(capture['size'])is not int or capture['size']!=child['training_size']:raise ValueError('exact captured training child')
     captured[slot]=capture
    deferred=row.get('training_document_deferred_slots',[])
    if type(deferred)is not list or any(type(slot)is not int for slot in deferred)or len(set(deferred))!=len(deferred)or set(deferred)!=set(children)-set(captured):raise ValueError('exact deferred capture partition')
   for child in p['batches']:
    if captured is not None and child['slot']not in captured:continue
    key=(miner,child['slot'])
    if key in committed:raise ValueError('duplicate declared slot')
    committed[key]=(row,child);uid=identity_uids.get(miner)
    if type(uid)is int and 0<=uid<256:grid[uid]+=1
    else:outside+=1
  # Eligibility is the entire authenticated committed-child inventory; bounded
  # training submissions contain only a selected subset of those documents.
  candidates={}
  for key,(_,child) in committed.items():
   child_key=(child['training_sha256'],child['training_size'])
   candidates.setdefault(child_key,[]).append(key)
  eligible=set();full_inventory={}
  for row in population['eligible_inventory']:
   if set(row)!={'learner_admission_sha256','sha256','size'}or type(row['size'])is not int or row['size']<=0:raise ValueError('eligible inventory shape')
   admission_digest=row['learner_admission_sha256']
   if not isinstance(admission_digest,str)or len(admission_digest)!=64 or any(c not in '0123456789abcdef'for c in admission_digest):raise ValueError('eligible admission digest')
   matches=candidates.get((row['sha256'],row['size']),[])
   if len(matches)!=1 or matches[0]in eligible or admission_digest in full_inventory:raise ValueError('unique exact eligible signed child')
   key=matches[0];eligible.add(key);full_inventory[admission_digest]=row;uid=identity_uids.get(key[0])
   if type(uid)is int and 0<=uid<256:eligible_grid[uid]+=1
  selected=set();inventory=[]
  for row in document['submissions']:
   signed=row['learner_admission'];p=authenticated(signed,authority);key=(p['miner_identity'],p['slot'])
   if key in selected or key not in eligible or p['version']!='committed-unaudited-training-v1'or p['assurance']!='unaudited'or p['epoch']!=epoch or p['checkpoint']!=checkpoint or p['source_sha256']!=source:raise ValueError('eligible admission scope')
   original,child=committed[key]
   if p['commitment_sha256']!=original['commitment_sha256']or digest(p['original_commitment'])!=original['commitment_sha256']or p['batch_sha256']!=child['batch_sha256']or p['proof_sha256']!=child['sha256']or p['document_sha256']!=child['training_sha256']or row['sha256']!=p['document_sha256']or row['size']!=p['document_size']or p['document_size']!=child['training_size']:raise ValueError('exact declared child admission')
   item={'learner_admission_sha256':digest(signed),'sha256':row['sha256'],'size':row['size']}
   if full_inventory.get(item['learner_admission_sha256'])!=item:raise ValueError('selected admission is exact eligible subset')
   inventory.append(item);selected.add(key)
  # Collection counts structurally admissible candidates, while captured
  # inventory also includes malformed documents. Keep those visible as excluded.
  structural=set()
  for exclusion in population.get('exclusions',[]):
   if exclusion.get('reason')!='structural_ineligible':continue
   sha=exclusion.get('document_sha256')
   matches=[key for key,(_,child)in committed.items()if child['training_sha256']==sha]
   if len(matches)!=1 or matches[0]in structural or matches[0]in eligible:raise ValueError('exact structural exclusion')
   structural.add(matches[0])
  if population['committed_count']!=len(committed)-len(structural)or population['eligible_count']!=len(eligible)or population.get('training_count',len(eligible))!=len(selected):raise ValueError('actual inventory counts')
  if 'training_selection'in population:
   selection=population['training_selection']
   if (selection['version']!='bounded-postfreeze-learner-selection-v1'or selection['eligible_count']!=len(eligible)or selection['training_count']!=len(selected)or selection['unselected_count']!=len(eligible)-len(selected)or selection['cap']!=256 or len(selected)>256 or selection['eligible_inventory_sha256']!=digest(population['eligible_inventory'])or selection['selected_inventory_sha256']!=digest(inventory)):raise ValueError('bounded training selection inventory')
  elif len(selected)!=len(eligible):raise ValueError('historical unbounded projection requires complete admitted inventory')
  return dict(submitted=len(committed),learner_training_selected=len(selected),learner_eligible=len(eligible),learner_excluded=len(committed)-len(eligible),submitted_grid=grid,eligible_grid=eligible_grid,unassigned=outside,submitting_identities=len({key[0]for key in committed}),eligible_identities=len({key[0]for key in eligible}),input_assurance='unaudited',proof_verification_claimed=False,source='authenticated-committed-learner-population',provenance_sha256=digest(document))
 except Exception:
  # Public state never contains the submitted documents, capabilities, or traces.
  # Invalid projection is unavailable; never silently invent a zero population.
  return None
