# Reproducing the controlled SQL diversity source

The published checkout preserves the source used by earlier experiments. The
reviewed SQL diversity dispatcher and public-question proposal policy are prepared
in a **new source directory**, without changing that checkout or active services:

```bash
.venv/bin/python -m ops.prepare_native_sql_diversity_source \
  --source /path/to/affine \
  --destination /path/outside/affine/sql-diversity-source
```

Use a clean committed Git checkout without untracked package/vendor files.
Source or destination aliases through symlinks are rejected.

The utility verifies eight approved source-file hashes before copying `subnet/`
and `prototype/vendor/`. It patches only the environment dispatcher, harness and
GPU runtime. Its default output for those three files is byte-identical to the
qualified `c454ddb7…` diversity worker. It writes `source-preparation.json` with
input/output hashes. A changed base source or existing destination is rejected.

Models, wallet keys, private task collections, databases, state and Git history
are not copied. The operator must independently provide qualified public actor
images and private original Spider tasks at
`/root/native-sql-common-v1/operator/private-tasks.json`, as described in
[NATIVE_SQL_DEPLOYMENT.md](NATIVE_SQL_DEPLOYMENT.md). An alternate absolute
`--private-tasks` path changes source identity and requires fresh source/spec
pins, an immutable worker bundle and runtime qualification before use.

The policy proposes SQL from the original public question/schema, then performs
model-conditioned candidate selection and the original bash-tool and private
native-grader flow. This is a controlled cohost experiment, not unrestricted
sampling or proof that an untrusted host cannot access operator resources.
Source preparation alone does not execute inference, replay, training or chain
transactions. Existing evidence is not reclassified when a new source is built.
