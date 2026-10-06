A ROOT-scoped CPU operator projection can recover a publication request that
failed before any runner marker because it inherited a train-only recovery
declaration. It removes only `training_startup_recovery` for a fresh upload
manifest and fresh reservation label. The original request, failure, completed
training and independent optimizer readback are retained. All sealed source
files, ten checkpoint files and original PUT capabilities stay unchanged.

`authorize` authenticates the exact original signed request and ROOT-signed
projection scope. The operator must additionally authenticate the scope-pinned
original completion/readback files and fresh physical absence witness before
launch. The helper is installed explicitly with
`install_remote_jobs_projection(RemoteJobs, *authorized_values)` before the
unchanged sealed gpu_service is run with `--once`. The unchanged checkpoint
staging path independently GETs and SHA-verifies all ten objects before ROOT
publication. No new training or optimizer step is authorized.

The scope expires no later than the original capability deadline. A later
recovery requires fresh explicit capability authorization, not backdating.
This option is off unless explicitly installed; it is not a science source
change or a claim that the original invalid upload request passed validation.
