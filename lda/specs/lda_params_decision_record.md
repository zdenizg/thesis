# LdaMulticore / alpha='auto' Incompatibility — Decision Record

## Discovered issue

Gensim's `LdaMulticore` raises `NotImplementedError` when `alpha='auto'`
is passed.  The asymmetric learned-alpha optimisation is only implemented
in `LdaModel` (single-process).  This means the pre-registered spec
(`lda_params.md`) specifies two parameters that are mutually exclusive in
Gensim 4.4.0:

- `alpha='auto'` — requires `LdaModel`
- `workers=3` — requires `LdaMulticore`

## Option A — Keep alpha='auto', switch to LdaModel

- **Change:** Use `gensim.models.LdaModel` instead of `LdaMulticore`.
  Drop the `workers=3` parameter (not accepted by `LdaModel`).
- **Methodology impact:** The learned asymmetric alpha prior is preserved
  exactly as pre-registered.  The only loss is multi-core parallelism
  during training — the mathematical model is identical.
- **Runtime estimate:** Based on the partial run (k=5 trained in 19s,
  k=10 in 25s with `LdaModel`), each k takes roughly 20–40s to train
  plus 13–20s for c_v coherence.  Full 10-value broad sweep: ~6–10
  minutes.  Fine sweep (perhaps 15 additional k values): ~10–15 minutes.
  Total wall time for both sweeps: under 30 minutes.

## Option B — Keep LdaMulticore, change alpha to 'symmetric'

- **Change:** Set `alpha='symmetric'` (or a fixed float like `1/k`) to
  allow `LdaMulticore` with `workers=3`.
- **Methodology impact:** Departs from the pre-registered spec.  Symmetric
  alpha assumes all topics are equally prevalent a priori, which may be
  inappropriate for a corpus where some topics (e.g., administrative
  boilerplate) are expected to dominate.  The learned asymmetric prior
  adapts to the actual topic distribution.
- **Runtime estimate:** Multi-core training would be ~2–3x faster per k,
  but coherence computation (the bottleneck) is unaffected.  Net saving
  is modest — perhaps 5–10 minutes across both sweeps.

## Decision

**Option A chosen.**  The asymmetric learned alpha is a substantive part
of the registered model — it allows the prior to reflect the fact that
some topics (e.g., administrative boilerplate) are inherently more
prevalent than others.  The `workers` parameter is a computational
detail that does not affect the mathematical model.  The runtime cost of
single-threaded training is approximately 5–10 additional minutes across
both sweeps, which is negligible.  `lda_params.md` has been updated to
remove `workers` and document this decision via footnote.

## Implementation detail: update_every=3

`LdaMulticore` with `workers=3` and `chunksize=2000` performs an M-step
after accumulating statistics from `workers × chunksize = 6000`
documents.  For a 4,049-document corpus this means one M-step per pass.
`LdaModel` defaults to `update_every=1`, which would M-step after every
2,000-document chunk — roughly two M-steps per pass — changing the
optimisation trajectory.  Setting `update_every=3` on `LdaModel`
replicates the pre-registered multicore M-step schedule as closely as
possible, maintaining methodological fidelity to the pre-registered
intent.
