# Fresh-source persistent trainer startup

The E13 original trainer failed before runtime/model loading because admission
imported `persistent_publication.validate_policy`, then the fresh-source guard
treated that same pure admission module as a preloaded compute implementation.
The original job's terminal failure and optimizer parent remain unchanged.

When publication is explicitly part of the authenticated execution inventory,
the loader now evicts its bootstrap import and reloads its implementation from
the pinned source bytes. Source hash validation still precedes eviction, and a
preloaded model runtime still fails closed. This does not authorize executing a
different source under the failed original request or deleting its job record.

Four regression controls cover fresh pinned reload, runtime preload rejection,
no eviction without explicit admission, and hash rejection before eviction.
The actual original signed E13 admission also passes the corrected CPU-only
bootstrap reproducer. These checks do not establish a completed training step;
deployment and authenticated startup-failure recovery remain separate.
