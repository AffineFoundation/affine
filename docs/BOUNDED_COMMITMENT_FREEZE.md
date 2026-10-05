# Bounded small commitment freeze

Only signed hourly commitment epochs use a dedicated small-object GET client:
5-second connect timeout, 10-second socket read timeout, and one total request
attempt. It preserves the existing storage endpoint, credentials, TLS defaults,
and signing config. Shared heavy-artifact/checkpoint clients remain unchanged.
At most four commitment GETs overlap; each reads at most 65,537 bytes. Journal
writes, signature/context checks, artifact HEAD checks, conditional copies and
receipt publication remain serial. Successful siblings and the first journaled
commitment survive retries. A failed public receipt PUT retries from the exact
frozen journal, without fetching or copying mutable objects again.

The executor drains started requests rather than pretending they were stopped.
Socket timeouts and one attempt bound retry exposure, but DNS/OS scheduling and
HEAD/copy/publication operations mean the signed freeze cutoff is an admission
and observation cutoff, not a proven absolute wall-clock limit. Transport
failures retry or defer as infrastructure, never fraudulent or missing samples.
A measured one-miner freeze does not qualify a 256-miner 60-second freeze.
