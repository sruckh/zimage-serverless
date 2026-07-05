# Decision Log — Z-Image Image Quality Improvement

This file records, for every image-quality-affecting change: the parameter,
old → new value, a cited source, and why the source supports the change. It
also records the live before/after RunPod endpoint test (best-effort,
non-blocking).

## Live endpoint test status

- **2026-07-04**: Submitted the "before" job (fixed test prompt, `seed=42`,
  `TEST_LORA_URL` LoRA) to the deployed endpoint (`.goals/live_test.py
  before`). Job `261d3b25-6db4-48ed-b9d0-fd6b6babee3a-u1` sat `IN_QUEUE` for
  over 10 minutes with no progress. Endpoint diagnostics
  (`.goals/check_endpoint.py`) showed both existing workers with
  `desiredStatus: EXITED` and no worker scaling up to pick up the job. This
  looks like a real endpoint/infrastructure availability issue, independent
  of `handler.py`. Per the goal's non-blocking rule for live testing, this
  does not block the required success criteria — implementation, lint,
  review, and README work proceed regardless. Will retry the before/after
  capture later in this run.
- **Security note**: `GET /v1/endpoints/{id}?includeWorkers=true` returns
  each worker's full environment block in plaintext, including secrets
  (HF/CivitAI tokens, RunPod internal API key, S3 credentials). Reported to
  the user directly; `.goals/check_endpoint.py` now redacts worker `env`
  from its output entirely.
- **Operational observation (independent of this change)**: while the first
  before-job attempt was queued, endpoint diagnostics showed 5 different
  worker instances spin up and exit (`desiredStatus: EXITED`) within minutes
  of each other, all still on the pre-change image (`4f5def1f6`). This looks
  like it could be a startup crash-loop or cold-start instability unrelated
  to the code change in this run — flagged here for visibility, not
  something this goal's scope includes fixing.
- **2026-07-05**: Resubmitted the before-job (identical fixed prompt,
  `seed=42`, `TEST_LORA_URL`). Job `af6c878e-982c-4922-9e3b-dcb5a3f6243b-u2`
  completed on image build **86** (pre-change, `4f5def1f6`) —
  `delayTime=13.5s`, `executionTime=213s`. Output image downloaded and saved
  to `.goals/live_test_before.json` (job record) and reviewed directly.
  Result: a coherent, on-prompt street-fashion portrait — beige knit dress,
  brown suede over-the-knee boots, structured handbag, concrete wall
  background all present and correctly composed — but overall softness:
  the knit's individual stitch definition (explicitly requested in the
  prompt) is barely visible, and fine texture (suede nap, fabric weave)
  reads as smooth/flat rather than crisp. Consistent with the reported "not
  crisp" complaint and with `shift=1.0` being far off both checkpoints'
  calibrated value.
- After `git push` (commit `0d8d601`), RunPod's GitHub integration built a
  new image automatically; within a few minutes one worker showed
  `imageName` ending in `0d8d60194` (matching the new commit) — confirmed
  via `.goals/check_endpoint.py`. Forced a rolling release with
  `.goals/force_rolling_release.py` (`PATCH
  https://rest.runpod.io/v1/endpoints/j7rrb3raom3lzh`, no-op body) — returned
  `200`. Submitted the after-job with the identical job input immediately
  after.
- Job `a88e05b3-74f4-462d-8b59-e949e0f66abc-u1` completed on `workerId
  3e65fpr72c5l53` — confirmed via `.goals/check_endpoint.py` to be running
  image tag `0d8d60194` (the new commit), not the pre-change `4f5def1f6`.
  Record saved to `.goals/live_test_after.json`.
- **Visual comparison (before v86 `shift=1.0` vs. after v87 `shift=6.0`,
  identical prompt/seed/LoRA)**: composition, pose, and framing are
  consistent between both (same seed), as expected. The "after" image shows
  a clear, direct improvement in exactly the areas the prompt stresses and
  the user complained about: the knit sweater's ribbed stitch structure is
  visibly more defined (individual ribs/columns are distinct, where the
  "before" reads as a flatter, smoother knit), and the suede boots show more
  texture/nap definition and visible creasing at the knee. The face also
  reads slightly sharper. This is consistent with the grounded hypothesis:
  the checkpoint-native `shift=6.0` redistributes inference steps toward
  the detail-refinement portion of the schedule relative to the previous
  hardcoded `shift=1.0`, producing crisper fine texture without changing
  composition (same seed → same coarse structure). This is one sample at
  one seed/prompt, not a statistical claim — but it directly corroborates
  the citation-based reasoning above with an actual before/after result on
  the live deployed endpoint.

## Quality-affecting parameter changes

### 1. `shift` default: `1.0` (both variants) → `6.0` (Base) / `3.0` (Turbo)

- **Old**: `handler.py` defaulted `shift=1.0` for both `Tongyi-MAI/Z-Image` and
  `Tongyi-MAI/Z-Image-Turbo`, with a comment claiming this "matches the
  official Tongyi-MAI recommended inference settings."
- **New**: when the caller doesn't specify `shift`, it's left `None` so
  `_configure_scheduler`'s own fallback (`pipeline.scheduler.config`'s
  already-loaded value) is used unchanged — i.e. whatever shift the loaded
  checkpoint actually ships with. In practice, for the two checkpoints that
  exist today, that resolves to `6.0` for Base / `3.0` for Turbo (see
  sources below) — but expressed as "use the checkpoint's own config"
  rather than hardcoded Python literals, so it can't drift if a future
  checkpoint/finetune revision ships a different value. (A code-review pass
  on this change caught the hardcoded-literal version as a fragile bandaid
  before this was finalized — see "Code review" section below.)
- **Source**: the actual `scheduler_config.json` shipped with each checkpoint
  on Hugging Face (fetched directly, not assumed):
  - `Tongyi-MAI/Z-Image`: `{"_class_name": "FlowMatchEulerDiscreteScheduler",
    "use_dynamic_shifting": false, "shift": 6.0}` —
    https://huggingface.co/Tongyi-MAI/Z-Image/blob/main/scheduler/scheduler_config.json
  - `Tongyi-MAI/Z-Image-Turbo`: same class, `"shift": 3.0` —
    https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/blob/main/scheduler/scheduler_config.json
  - Neither the official GitHub README
    (https://github.com/Tongyi-MAI/Z-Image, raw README fetched directly) nor
    the HF model card (https://huggingface.co/Tongyi-MAI/Z-Image) mentions
    `shift` anywhere in "Recommended Parameters" — so the prior "matches the
    official recommendation" claim in the code comment had no actual source
    backing it.
- **Why this matters**: `use_dynamic_shifting: false` means these are static,
  checkpoint-calibrated schedule shifts, not something the model
  auto-derives — every request was previously running with a shift 3-6x
  lower than the value the checkpoint was actually trained/calibrated
  against, on every single call (`_configure_scheduler` is invoked
  unconditionally since `shift` was never `None`). A flow-matching schedule
  shift mismatch of this size directly changes how inference steps are
  distributed between coarse structure and fine detail, which is consistent
  with the reported "lines not crisp, lots of artifacts" symptom.
- **Caveat honestly recorded**: the default checkpoint actually loaded
  (`famegridZIB_v10.safetensors`, injected into the transformer at load time)
  is a community finetune of Z-Image Base, not the stock Tongyi-MAI weights.
  I looked for finetune-specific recommended settings
  (https://civitai.com/models/2533927/famegrid-zib-checkpoint) but the page
  failed to render server-side content (scrape returned only a client-side
  error shell) — no finetune-specific override was found, cited, or assumed.
  Flow-matching finetunes conventionally preserve the base schedule unless
  documented otherwise, so the base checkpoint's own shipped value (6.0) is
  used as the grounded default; this is flagged here rather than silently
  assumed.

### 2. `cfg_normalization` default: unchanged (`True`), citation tightened

- **Old/New**: still defaults to `True` — no value change.
- **Source**: official GitHub README, "Recommended Parameters": *"CFG
  normalization: `False` for general stylism, `True` for realism."* —
  https://github.com/Tongyi-MAI/Z-Image (raw README fetched directly).
- **Why**: this worker targets photorealistic output (negative prompt list,
  fixed test prompt are both photography-oriented), so `True` is the correct
  default per the README's explicit guidance — even though the README's own
  code sample uses `cfg_normalization=False` for a same-page example. That
  contradiction is called out explicitly in the code comment now, rather
  than glossed over, since presenting the sample code as if it were the
  "recommendation" would have been the wrong citation.

### 3. Second-pass (hires-fix) strength/rationale: unchanged, citation added

- **Old/New**: `second_pass_strength=0.42` unchanged — no value change.
- **Source**: https://github.com/Tongyi-MAI/Z-Image/issues/144 — multiple
  users confirm Z-Image Base output looks "never fully denoised"/blurry
  regardless of sampler, and the documented community workaround is exactly
  what this second pass already does: "use Z Image Turbo on a second sampler
  with a very very light denoise which allows to 'clean' the image."
  Previously the code comment referenced "issue #144" without a URL; added
  for verifiability.

## Not changed, and why

- `guidance_scale` (Base default `4.5`) and `steps` (Base default `50`) both
  fall inside the official README's recommended ranges (guidance 3.0–5.0,
  steps 28–50), so no change was made — moving either would trade one
  unverified assumption for another without a citation to justify a specific
  new value.
- VAE forced to `float32` for decoding: this is standard practice to avoid
  bf16 VAE banding/pixelation, but no Z-Image-specific citation was found
  confirming or refuting it for this model; left unchanged since there is no
  grounded reason to touch it.

## Code review

Ran a two-angle review pass (correctness; cleanup/altitude/conventions) over
the `handler.py` + `.flake8` diff via independent sub-agents.

- **Correctness angle**: no bugs found. Confirmed `is_turbo` is computed
  before `shift`, both `_configure_scheduler` call sites (base + img2img
  pipeline) consume the same `shift` value consistently, all the autopep8
  single-line-to-multi-line reformattings (the `vae_tiling` default, the
  `lora_entries` comprehension, the `if/else` and `os.remove` splits)
  preserve exact original semantics/indentation, and the removed
  `as_completed` import was genuinely unused (code uses `pool.map`).
- **Cleanup/altitude angle**: flagged that hardcoding `3.0 if is_turbo else
  6.0` as Python literals was a fragile bandaid — `_configure_scheduler`
  already reads the live-loaded checkpoint's own `scheduler.config["shift"]`
  as its fallback, so hardcoding literals meant the fix wouldn't track a
  future checkpoint/finetune revision that ships a different shift.
  **Fixed**: `shift` is now left `None` when not explicitly requested in the
  job input, so `_configure_scheduler`'s existing fallback path is used
  unchanged — the checkpoint's own shipped config is the source of truth,
  not a hardcoded number. No CLAUDE.md conventions violations found.

## README ↔ handler.py cross-check

Enumerated every `job_input.get("<key>", ...)` in `handler.py` (30 total) via
`grep -oE 'job_input\.get\("[a-zA-Z_0-9]+"[^)]*\)' handler.py`, and every
`os.environ.get("<VAR>", ...)` (8 total). Confirmed each of the 30 job-input
keys appears in README.md's "Endpoint Input Parameters" tables, and
cross-checked each documented default against the actual resolved default in
code:

| Parameter | handler.py default | README default | Match |
|---|---|---|---|
| `lora_scale` | `0.85` | `0.85` | yes |
| `width` | `1024` | `1024` | yes |
| `height` | `1024` | `1024` | yes |
| `steps` | `9` turbo / `50` base | auto → `50`/`9` | yes |
| `guidance_scale` | `0.0` turbo / `4.5` base | auto → `4.5`/`0.0` | yes |
| `cfg_normalization` | `True` | `true` | yes |
| `cfg_truncation` | `1.0` | `1.0` | yes |
| `max_sequence_length` | `512` | `512` | yes |
| `seed` | `42` | `42` | yes |
| `use_beta_sigmas` | `False` | `false` | yes |
| `shift` | `None` → checkpoint native (`6.0`/`3.0`) | checkpoint default `6.0`/`3.0` | yes — **fixed** (was stale `1.0` in 3 places, see below) |
| `upscale_model` | `nomos_webphoto` | `nomos_webphoto` | yes |
| `upscale_enabled` | `True` (env-overridable) | `true` | yes |
| `upscale_factor` | `1.5` | `1.5` | yes |
| `vae_tiling` | auto by resolution (`>1024×1024`) | auto | yes |
| `second_pass_enabled` | `True` (env-overridable) | `true` | yes |
| `second_pass_upscale` | `1.25` | `1.25` | yes |
| `second_pass_strength` | `0.42` | `0.42` | yes |
| `second_pass_steps` | `28` | `28` | yes |
| `second_pass_guidance_scale` | `4.5` | `4.5` | yes |
| `second_pass_seed` | `= seed` | `seed` | yes |
| `second_pass_cfg_normalization` | `True` | `true` | yes |
| `second_pass_cfg_truncation` | `1.0` | `1.0` | yes |
| `second_pass_max_sequence_length` | `= max_sequence_length` | `max_sequence_length` | yes — **fixed** (was stale literal `512`) |
| `second_pass_use_beta_sigmas` | `= use_beta_sigmas` | `use_beta_sigmas` | yes |
| `second_pass_vae_tiling` | `False` | `false` | yes |
| `second_pass_vae_slicing` | `True` | `true` | yes |
| `negative_prompt` | long photorealism-oriented string | *(was "see below" with no such section)* | yes — **fixed**, full text added |

`prompt`, `loras`, `lora_url` have no default (required or absent-by-default)
and are documented as such in both places — no mismatch possible.

**Mismatches found and fixed** (3): the `shift` default was documented as a
stale `1.0` in the feature-bullet list, the parameter table, and the
Scheduler Notes table — all three updated to describe the new
checkpoint-native default. `second_pass_max_sequence_length`'s README row
claimed a literal default of `512`, but the code actually defaults it to
whatever `max_sequence_length` resolved to (only `512` when that's also at
its own default) — fixed to say `max_sequence_length`. The `negative_prompt`
row pointed to a "see below" section that didn't exist — the actual default
string was added.

All 8 `os.environ.get(...)` calls in `handler.py` (`MODEL_ID`, `HF_TOKEN`,
`UPSCALE_DEFAULT_ENABLED`, `SECOND_PASS_DEFAULT_ENABLED`,
`UPSCALE_DEFAULT_MODEL`, `UPSCALE_DIR`, `UPSCALE_USE_CUDA`, `TORCH_COMPILE`)
were checked against the README's "Environment Variables" table — all
defaults matched with no changes needed. (`UPSCALE_USE_CUDA`'s effective
default was confirmed by reading `get_upscaler`'s actual usage:
`_to_bool(UPSCALE_USE_CUDA_ENV, default=False) and torch.cuda.is_available()`
— matches README's documented default of `false`.)
