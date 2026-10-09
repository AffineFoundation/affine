"""Idempotent CPU population admission without changing scientific contracts.

Every invocation authenticates the submitted authority signature. Full semantic
validation runs once per identical population in each process. Existing equal
populations need no durable rewrite; new populations retain original persistence.
"""
import hashlib
import json


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def install(service):
    original = service.ContinuousAuditor.admit
    authenticate = service.authenticate
    marker = '_authenticated_population_admission_cache_v1'
    if getattr(original, marker, False):
        return

    def admit(self, document):
        # Do not treat a local cache hit as authentication of miner/operator input.
        payload = authenticate(document, self.controller.authority.id)
        epoch = payload.get('manifest_document', {}).get('payload', {}).get('epoch')
        if not isinstance(epoch, str):
            return original(self, document)
        identity = hashlib.sha256(canonical(document)).hexdigest()
        existing = self.state['populations'].get(epoch)
        unchanged = existing is not None and existing == document
        checked = self.__dict__.setdefault(marker, {})
        if unchanged and checked.get(epoch) == identity:
            return None
        if not unchanged:
            result = original(self, document)
        else:
            # Original admission still checks canonical population, nested
            # signatures, bindings and eligibility. Persist would rewrite the
            # entire history with an identical value, so suppress only that call.
            had_override = 'persist' in self.__dict__
            previous_override = self.__dict__.get('persist')
            self.persist = lambda: None
            try:
                result = original(self, document)
            finally:
                if had_override:
                    self.persist = previous_override
                else:
                    del self.__dict__['persist']
        checked[epoch] = identity
        return result

    setattr(admit, marker, True)
    service.ContinuousAuditor.admit = admit
