# Decision Log — Z-Image Image Quality Improvement

This file records, for every image-quality-affecting change: the parameter,
old → new value, a cited source, and why the source supports the change. It
also records the live before/after RunPod endpoint test (best-effort,
non-blocking).

## Flash Attention: two bugs, found and fixed sequentially

1. **`handler.py` never actually enabled it** (fixed first). `attn_implementation="flash_attention_2"`
   passed to `ZImagePipeline.from_pretrained(...)` is not a recognized kwarg —
   diffusers silently drops it with a warning rather than raising, so the old
   try/except always "succeeded" and printed "Model loaded with Flash
   Attention 2" despite it never being enabled. Confirmed directly from a
   production log line: `Keyword arguments {'attn_implementation':
   'flash_attention_2'} are not expected by ZImagePipeline and will be
   ignored.` immediately followed by the false-positive success message.
   Fixed to call `pipe.transformer.set_attention_backend("flash")` after
   loading, per the official Tongyi-MAI/Z-Image README's documented Quick
   Start usage. Corrected the same claim in README.md.
2. **`runpod_bootstrap.sh`'s install line itself was buggy**, uncovered
   immediately after fix #1 made failures visible instead of silently
   swallowed: a fresh worker logged `RuntimeError: Flash Attention backend
   'flash' is not usable because of missing package or the version is too
   old.` The install line was:
   `pip install "$FLASH_ATTN_URL" ... || echo "..." && pip install flash-attn ...`
   — bash parses `A || B && C` as `(A || B) && C`, so **C (the slow,
   unpinned source-build fallback) runs unconditionally**, even when the
   fast prebuilt-wheel install (A) already succeeded. Confirmed directly:
   `bash -c 'true || echo b && echo c'` prints `c` even though `true`
   succeeded. This redundant fallback build can fail or produce a broken
   install that clobbers the wheel install that had just worked. Fixed with
   an explicit `if/else` so the source fallback only runs when the wheel
   install actually fails, and added a post-install `import flash_attn`
   verification line (logs a clear warning if still not importable, instead
   of finding out indirectly at inference time).
   - This fix lives inside the bootstrap's first-run gate (`.installed_v3`),
     which had already tripped on existing worker volumes — so the fix alone
     wouldn't run for them. Bumped `INSTALL_FLAG` to `.installed_v4` to force
     re-triggering on the next cold start.
   - Source: `https://github.com/Tongyi-MAI/Z-Image` (Quick Start section,
     `set_attention_backend` usage, fetched directly earlier this session).

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
- **2026-07-05, LoRA fix verification**: after the LoRA-loading fix (commit
  `80b074e`, see "Quality-affecting parameter changes" #4 below), pushed,
  waited for the new image (`80b074e6b`) to appear on two workers, forced
  another rolling release, and resubmitted the identical fixed-prompt job.
  Job `13253068-fcf6-4444-9efa-da305daa14e2-u2` completed on `workerId
  ikaho6mpenuawv`, confirmed running image `80b074e6b`. Result, cropped and
  zoomed on the face exactly as before/after v86/v87: a dramatic, unambiguous
  improvement — real skin texture (visible pores, natural tonal variation)
  in place of the flat waxy/plastic look present in *both* prior samples,
  coherent anatomical structure with no warping, and a visibly distinct
  identity consistent with the character LoRA actually contributing signal
  now (rather than the generic, LoRA-not-really-applied face seen before the
  fix). This directly corroborates the root-cause finding: the LoRA's
  weights were being silently dropped/corrupted before, and are now loading
  and applying correctly. User's original complaint ("the face... is warped
  and distorted... looks like putty") addressed by this fix, independent of
  the earlier `shift` fix (which only affected fabric/background texture).

  **Correction (same day, user pushback):** the above claim was overstated.
  The test prompt used for this comparison (`.goals/live_test.py`'s fixed
  texture-stress prompt) never included the LoRA's trigger word (`K1mScum`)
  and never referenced the actual character's appearance — so "looks more
  coherent/less waxy" was true, but is not evidence the LoRA's *identity*
  was rendering correctly. It wasn't a valid identity test. See the next
  section for the real investigation this prompted.

### LoRA identity-match investigation (post-fix), prompted by user feedback

After the fix above, the user tested with a proper identity prompt (trigger
word included: `"K1mScum, A poised middle-aged woman with..."`) and reported
the result "does not look like the character... not even a distant cousin
... less than 50% matching," and raised the concern that the LoRA might
still not be applying correctly. Investigated with controlled live tests
against the deployed endpoint (same seed=42, same prompt, `second_pass_enabled=true`,
width=864/height=1152), each confirmed via `workerId` → image tag to have
run on the LoRA-fix build:

1. **Reused the texture-test prompt's read as a positive signal was wrong.**
   Re-examined: hair color/eye color matching between generations and the
   reference is not proof the LoRA is contributing anything, since the
   prompt text itself explicitly describes "chestnut-brown hair" / "warm
   hazel eyes" — the base model renders that from the text alone, LoRA or
   not. Only facial *structure/identity* (which text can't specify) is a
   valid signal, and that's what the user correctly flagged as not matching.
2. **`scale=1.2` vs. no-LoRA-at-all control, identical seed/prompt:** the two
   results were visually indistinguishable. This confirmed the user's
   concern directly — at that scale, the LoRA contributes ~nothing
   distinguishable from the base model's own output.
3. **Root cause: a second, independent scaling stage was already in the
   code, uninvolved in the earlier bug.** `_activate_loras` (handler.py)
   calls `pipeline.set_adapters(adapter_names, adapter_weights=[scale])`
   after loading — diffusers' own standard LoRA-strength control, applied
   *on top of* whatever's baked into the weights. So effective scale =
   `(alpha/rank baked in) × (set_adapters weight)`. Before the earlier fix:
   baked ≈1.0 (the bug) × 0.85 (request default) ≈ 0.85 effective — moot,
   since those keys were being dropped entirely (0%). After the fix: baked
   0.5 (this LoRA's actual trained ratio) × 0.85 ≈ 0.425 effective — and
   the `scale=1.2` test only reached ≈0.6. This is architecturally correct
   (matches how diffusers' own official Z-Image LoRA converter works: bake
   training-time alpha once, `set_adapters` is the single further user
   control, `1.0` = full designed strength) but means the nominal 0.85→1.2
   bump moved the *effective* scale far less than expected.
4. **`scale=2.5` (effective ≈1.25), same seed/prompt:** produced a real,
   visible change from the no-LoRA control (hair styling shifted, expression
   changed, a hint of nasolabial-fold/jaw definition appeared) — confirming
   the LoRA mechanism is genuinely responsive to scale and not silently
   inert — but still not a strong, unmistakable character match against the
   user's reference images (one of which, `Krea2_turbo_00001_jjjhe...png`,
   uses the *identical* prompt against a different base model trained on
   identical data — the most direct ground truth available).
5. **Checkpoint mismatch hypothesis, raised by the user.** `handler.py`'s
   `get_pipeline()` loads stock `MODEL_ID` (`Tongyi-MAI/Z-Image`, confirmed
   as this LoRA's own declared base model via its HF "Model tree" metadata —
   ruling out a Base/Turbo mismatch) but then **unconditionally overwrites
   the transformer's weights** with a third-party community finetune
   (`famegridZIB_v10.safetensors`, downloaded automatically by
   `runpod_bootstrap.sh` whenever `CIVITAI_TOKEN` is set — which it is on
   this endpoint) *before* any LoRA is applied. Confirmed via CivitAI's API
   (`https://civitai.com/api/v1/models/2533927`, since the page itself is
   JS-gated and un-scrapable) that this is a **full 11.46 GB checkpoint**
   (`"size": "full"`, bf16) — a complete parameter finetune, not a light
   merge — whose own stated goal ("natural-looking skin with some texture,
   not overly smoothed or plastic") is notably *not* what our outputs show
   either (they read smooth/plastic), suggesting a real, unresolved
   interaction rather than just a LoRA-strength tuning question. This
   substitution is **not** in the README's documented environment variables
   at all (a genuine gap — `CHECKPOINT_PATH`'s `os.environ.get(...)` call
   spans multiple lines, which is why the earlier single-line-regex
   README-cross-check in this log missed it entirely).
6. **Fix**: added `USE_CIVITAI_CHECKPOINT` (default `true`, preserves
   existing behavior) to both `handler.py` (`get_pipeline`, gates the
   checkpoint injection) and `runpod_bootstrap.sh` (gates the download
   itself, via a `case` statement matching the same falsy aliases as
   `_to_bool` — a code-review pass caught the first version's plain
   `[ "$VAR" = "false" ]` check as case-sensitive/narrower than `_to_bool`,
   which would let the two scripts disagree on "0"/"False"/"no"/"off").
   Documented `USE_CIVITAI_CHECKPOINT`, `CHECKPOINT_PATH`, and `CIVITAI_TOKEN`
   in README.md's environment variables table (closing the gap noted above).
   Lets the user test the LoRA against stock Base (set
   `USE_CIVITAI_CHECKPOINT=false` on the endpoint) without a code change per
   test, and is a reusable, permanent capability, not a one-off diagnostic
   hack.
   Sources: `https://huggingface.co/Gemneye/K1mScum-ZImage-Base` (Model tree:
   base model `Tongyi-MAI/Z-Image`); `https://civitai.com/api/v1/models/2533927`.

7. **Confirmed live, container build v94.** User set `USE_CIVITAI_CHECKPOINT=false`
   on the endpoint and reran the identical trigger-word prompt (`scale=0.85`,
   default — no scale boost this time). Logs confirm: `USE_CIVITAI_CHECKPOINT=false
   — using stock base model weights.` (checkpoint injection correctly skipped) and
   `Added 210 missing LoRA alpha keys (network_alpha=16.0)` (alpha fix reading the
   correct metadata value). Result: a substantial, unambiguous improvement over
   every prior test — visible natural forehead lines and skin texture (present in
   the Krea2/photo references, absent from every famegridZIB-based test regardless
   of LoRA scale), matching chestnut hair and hazel-amber eyes, and a plausibly
   recognizable version of the same person. Not a perfect match (eye color reads
   slightly warmer/browner than the reference's green-hazel, face slightly rounder
   in the cheeks, and this particular generation didn't render the requested
   blazer) — but this is the closest result of the entire investigation, at the
   LoRA's default scale (0.85), with no scale boosting needed. **This confirms the
   checkpoint-mismatch hypothesis**: the famegridZIB finetune was the dominant
   factor suppressing this LoRA's identity signal, not the alpha-scaling bug or
   LoRA strength — those fixes were real and necessary, but this was the change
   that actually made the difference visible.
   - Log gap worth noting for future debugging: ~71s elapsed between "Added 210
     missing LoRA alpha keys" (20:10:36) and the first sampling step (20:11:47)
     with no intervening log lines in what the user pasted — likely covers
     Attempt 2's retry, Attempt 3's conversion (`_convert_lora_to_diffusers`),
     and `load_lora_into_transformer`, but wasn't confirmed line-by-line since
     the excerpt may have been trimmed. Worth getting the full log next time
     this path is exercised, to confirm definitively whether Attempt 2 or
     Attempt 3 is what's actually succeeding for this LoRA shape in practice.
   - Default for `USE_CIVITAI_CHECKPOINT` was **not** changed (stays `true`) —
     this result is strong evidence for LoRA compatibility specifically, but
     changing the platform-wide default (famegridZIB was previously chosen
     for its own stated photorealism/social-media aesthetic) is a broader
     product decision than this investigation's scope; left for the user to
     decide with this evidence in hand.

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

### 4. LoRA loading: fixed silent key-drop and missing alpha scaling in the manual-conversion fallback

- **Trigger**: user reported that with `TEST_LORA_URL` (`Gemneye/K1mScum-ZImage-Base`,
  a character LoRA), the generated face was "warped and distorted... like putty" —
  visually confirmed by cropping and zooming the face region of both the before
  and after live-test images (both showed the same waxy/plastic skin and
  slightly warped structure, unaffected by the `shift` fix above, which only
  improved fabric/wall texture).
- **Root cause, traced end to end and confirmed against real sources**:
  1. This LoRA's actual key format (confirmed via a safetensors header range
     request, no full download needed) is
     `diffusion_model.layers.N.attention.{to_q,to_k,to_v,to_out.0}.lora_A/B.weight`
     and `diffusion_model.layers.N.feed_forward.{w1,w2,w3}.lora_A/B.weight` —
     420 tensors across 30 layers, **no per-tensor `.alpha` keys at all**;
     alpha lives only in the file's `__metadata__`: `ss_network_alpha: '16.0'`,
     `ss_network_dim: '32'` (i.e. an intended `alpha/rank = 0.5` scale).
  2. Diffusers' own built-in Z-Image LoRA converter
     (`_convert_non_diffusers_z_image_lora_to_diffusers` in
     `diffusers/loaders/lora_conversion_utils.py`, fetched and traced directly)
     is written for **underscore**-joined module paths
     (`layers_0_attention_to_q...`) and ends with `if len(state_dict) > 0:
     raise ValueError(f"state_dict\` should be empty at this point...")`.
     Tracing this LoRA's **dot**-separated native keys through that function's
     splitter shows they don't collapse back to a recognized suffix, so this
     exact `ValueError` fires — and this codebase's own `_load_lora` Attempt-1
     handler already special-cases the literal substring `"state_dict\` should
     be empty"` as recoverable, confirming this exact path is what's hit in
     production (not a hypothetical). This falls through Attempt 2 (same
     underlying key shape, same failure) into Attempt 3, this file's own
     manual `_convert_lora_to_diffusers`.
  3. Attempt 3 had **two compounding bugs**, both confirmed against the actual
     diffusers `ZImageTransformerBlock`/`FeedForward` source
     (`transformer_z_image.py`, fetched directly):
     - A key-rename step assumed the model exposes `self.attn`/`self.ffn` with
       `fc1`/`fc2` feed-forward linears. The real model uses `self.attention`,
       `self.feed_forward`, and a **3-matrix gated (SwiGLU) FFN literally named
       `self.w1`/`self.w2`/`self.w3`** (`w2(silu(w1(x)) * w3(x))`) — there is
       no `attn`/`ffn`/`fc1`/`fc2` anywhere. The rename was corrupting every
       attention/feed-forward key for this LoRA, and had no mapping for `w3`
       at all (only `w1`→`fc1`, `w2`→`fc2`), silently orphaning a third of
       every layer's FFN LoRA weights even before the corruption is
       considered. **Fixed**: removed the rename entirely — native
       `attention`/`feed_forward`/`w1`/`w2`/`w3` keys need no renaming, they
       already match the model 1:1.
     - Separately and more severely: **no code path in this function ever
       moved plain `layers.N.*` keys into the returned `converted_state_dict`
       at all** — only `transformer_blocks.`/`single_transformer_blocks.`
       prefixed keys were swept in. So this LoRA's weights weren't just
       mis-mapped, they were **silently dropped in their entirety** — the
       function returned an effectively-empty converted dict, meaning the
       character LoRA was very likely contributing near-zero identity signal.
       **Fixed**: added `layers.`, `noise_refiner.`, `context_refiner.` to the
       set of prefixes swept into `converted_state_dict`.
  4. Independently: nothing in `_convert_lora_to_diffusers` (any branch) ever
     applied alpha/rank scaling to the converted weights — `.alpha` tensors
     were explicitly excluded from the "unconverted leftovers" warning
     (implying the author knew they'd never be used) and then just dropped.
     PEFT's `load_lora_adapter` (the actual API `load_lora_into_transformer`
     calls) has no built-in concept of per-tensor `.alpha` keys — that's a
     Kohya/A1111-ecosystem convention, not a PEFT one; diffusers' own official
     Z-Image converter bakes `alpha/rank` directly into the weight values for
     exactly this reason. Without that, every LoRA reaching this manual
     path loaded at an implicit `alpha == rank` (scale `1.0`) regardless of
     what it was actually trained at — for this LoRA, double its intended
     `0.5` scale, a well-documented cause of overcooked/distorted output
     concentrated on the LoRA's own subject (faces especially, since identity
     is packed into a small region). **Fixed**: `_load_lora` now reads
     `ss_network_alpha` from the safetensors file's own metadata (via
     `safetensors.safe_open(...).metadata()`) and threads it through both
     `_patch_missing_lora_alphas` (benefits Attempt 2, for LoRAs using
     diffusers' expected underscore format that happen to omit alpha tensors)
     and `_convert_lora_to_diffusers` (benefits Attempt 3). A first version
     of the Attempt-3 fix scaled `lora_B` inline while iterating and popping
     from a shared dict — a code-review pass caught that this silently
     skipped the scale whenever a LoRA's key order listed `lora_A` before
     `lora_B` (the common case), since the sibling lookup would find its `A`
     key already popped. Fixed by moving the scaling to a single pass **after**
     all conversion branches finish, over the finalized `converted_state_dict`
     (verified by a second review pass: every branch's A/B key-naming lines up
     consistently, and scaling only `lora_B` by the full `alpha/rank` factor
     is mathematically equivalent to scaling the whole `lora_B @ lora_A`
     product, since scalar multiplication commutes through matrix
     multiplication).
  - Sources: safetensors header of
    `https://huggingface.co/Gemneye/K1mScum-ZImage-Base/resolve/main/K1mScum-000086-Z-Image-Base.safetensors`
    (fetched via HTTP range request, not a guess);
    `https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/models/transformers/transformer_z_image.py`;
    `https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/loaders/lora_conversion_utils.py`;
    `https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/loaders/lora_pipeline.py`.

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
- **Flagged but not fixed**: `_convert_lora_to_diffusers`'s generic mapping
  `"adaLN_modulation.0" -> "norm_out.linear"` (handler.py, step 5) looks
  suspicious by the same pattern as the fixed `attn`/`ffn` bug — per the
  actual `ZImageTransformerBlock` source, each block has its own
  `self.adaLN_modulation` (a per-block modulation network), and there's a
  separate `FinalLayer.adaLN_modulation` for the output layer only; neither
  is obviously named `norm_out.linear` in the per-block case. Not changed
  because no adaLN-format LoRA sample was available to verify the correct
  mapping against (unlike the attention/feed_forward fix, which was verified
  against this LoRA's own real keys) — changing it without a concrete case to
  check would be trading one unverified guess for another.

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
