# Goal: Improve Z-Image RunPod Serverless Image Quality

## Background (grounded in repo state as of 2026-07-04)

- `handler.py` (~800 lines) is the entire RunPod serverless worker: loads a
  Z-Image diffusers pipeline, applies dynamic multi-LoRA loading from URLs
  (`handler`, `_load_lora`, `_convert_lora_to_diffusers`, `download_lora`),
  runs an optional img2img "hires-fix" second pass, and an optional spandrel
  detail upscaler (`get_upscaler`, `upscale_image`, `UPSCALE_MODELS`).
- Recent commit history (`git log`) shows an active, somewhat trial-and-error
  tuning history on quality knobs: `shift` flip-flopped between 1.0 and 3.0,
  `cfg_normalization` added, steps changed 9→50, VAE forced to float32, hires-
  fix decoupled from the upscaler, selectable upscalers added via spandrel.
  This history is exactly the kind of "assumption-driven" tuning the user
  wants replaced with cited, verified reasoning.
- **This host cannot run the model directly** — no GPU. Code only executes
  after `git push` triggers RunPod's auto-build/deploy of the container. Per
  project memory, dependencies for the running container belong in the
  `Dockerfile` / `runpod_bootstrap.sh`, never installed on this host.
- The user can provide live-testing credentials for the **already-deployed**
  endpoint, so a gitignored `.env` (see `.gitignore`) has been created at the
  repo root with placeholders:
  - `RUNPOD_ENDPOINT_URL` — full sync endpoint URL, e.g.
    `https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync`
  - `RUNPOD_API_KEY` — RunPod dashboard API key
  - `TEST_LORA_URL` — a LoRA URL to exercise multi-LoRA loading in the test job
  The user fills in real values before/during the run. This makes the "call
  the live endpoint" idea concretely runnable, not just a research question.
- **No lint config exists yet** (`ruff.toml`/`pyproject.toml`/`.flake8`/
  `.pylintrc` all absent). `flake8` is not installed on this host or in the
  Dockerfile. The user chose **flake8** as the lint gate for this goal.
- **Redeploy mechanism, confirmed via context7 (RunPod's official docs):**
  after a `git push` triggers RunPod's auto-build, the user's manual habit is
  to delete the endpoint's existing workers and let `workersMin: 2` spin up
  fresh ones on the new image. The scriptable equivalent is
  `PATCH https://rest.runpod.io/v1/endpoints/{endpointId}` (`Authorization:
  Bearer <RUNPOD_API_KEY>`) — official docs: *"Updates an existing serverless
  endpoint. This operation triggers a rolling release on the endpoint."* Even
  a no-op body triggers it. Source:
  https://docs.runpod.io/api-reference/endpoints/PATCH/endpoints/endpointId
  (companion: `PATCH /v1/templates/{templateId}` triggers a rolling release
  for every endpoint on that template —
  https://docs.runpod.io/api-reference/templates/PATCH/templates/templateId).
  No documented public API was found for deleting a *single* worker
  individually (the `podStop`/`podTerminate` GraphQL mutations are for rented
  Pods, a different resource type, not serverless workers) — the rolling-
  release PATCH is the grounded substitute, not an assumption. This mechanism
  is **already researched and confirmed** — the loop should *use* it directly
  in the live-test script (Best-effort bonus, below) rather than re-researching
  whether a redeploy mechanism exists. Live testing itself remains
  non-blocking overall (no GPU/guaranteed credentials on this host), but if
  it does run, it should redeploy via this exact call, not skip straight to
  testing against a possibly-stale worker.

## Success criteria (all required, deterministic, evaluator-checkable)

1. **Lint clean.** A `.flake8` config exists at the repo root (tuned so a
   large single-file handler with intentional broad `except Exception:`
   fallback chains doesn't drown in noise — but not so loose it's a no-op).
   `flake8 .` exits `0`.
2. **Every quality-affecting change is cited.** Every parameter/default
   changed for image-quality reasons (scheduler `shift`, `cfg_normalization`,
   `cfg_truncation`, `steps`, `guidance_scale`, upscaler model/params,
   second-pass strength/steps/guidance, VAE dtype/tiling/slicing, sampler
   choice, LoRA-blending behavior) has an entry in `.goals/decision-log.md`
   with: the parameter, old → new value, a URL to a reliable source (official
   Tongyi-MAI/Z-Image repo or model card, HuggingFace/diffusers docs consulted
   via context7, or a firecrawl-verified GitHub issue/PR/technical writeup),
   and 1-3 sentences on why the source supports the change. No entry may be
   justified only by "should look better" / assumption.
3. **Code review completed.** A review pass (use the `code-review` skill or
   equivalent manual review) runs over the diff. Zero unresolved `CONFIRMED`
   correctness findings remain (fixed, or explicitly logged as
   `no_change_needed` with reasoning in the same decision log).
4. **README matches the code, exactly.** Every `job_input.get("<key>", ...)`
   default and every `os.environ.get("<VAR>", ...)` default in `handler.py`
   is documented in `README.md` with the *same* default value and an accurate
   description usable by a front-end developer with no access to the Python
   source. Write and run a small cross-check (script or manual line-by-line
   diff, shown in the final report) proving zero mismatches between
   `handler.py` defaults and the README's parameter tables.
5. **Syntactically valid.** `python3 -m py_compile handler.py` exits `0`
   (this repo has no GPU to run further, so this plus lint is the ceiling for
   automatic verification on this host).
6. **Scope respected.** `git diff --stat` touches only: `handler.py`,
   `Dockerfile`, `runpod_bootstrap.sh`, `requirements.txt`, `README.md`,
   `.flake8`, and files under `.goals/`.

## Fixed test prompt (for before/after live quality check)

Use this exact prompt, unmodified, for every live before/after call described
below — it's designed to stress fine texture (knit stitching, suede nap),
clean lines/edges (garment seams, bag hardware, wall geometry), skin detail,
and depth-of-field falloff, which is where the reported "lines not crisp,
lots of artifacts" problem shows up most:

```
A three-quarter street-style fashion portrait of woman in mid-stride along an urban sidewalk, her body angled slightly toward camera with a natural, confident walking cadence — weight shifting forward onto her leading foot, arms in easy motion at her sides. Her expression is composed and forward-focused, chin level, gaze directed just past the lens with the cool indifference of someone entirely at ease in her own presence.She wears a voluminous beige chunky-knit sweater dress that falls to mid-thigh — the open shoulder detail exposing one collarbone cleanly, the decorative lacing at the chest rendered with visible cord tension and eyelet hardware catching ambient light. The knit structure shows individual stitch definition, the fabric draping with realistic weight and slight swing from the motion of her stride, producing natural asymmetric folds along the hem. Her legs are fitted in tall, form-hugging brown suede over-the-knee boots with a structured block heel — the suede surface showing its characteristic fine nap texture, subtle compression wrinkles behind the knee from movement, and a matte finish that absorbs rather than reflects the ambient light. Her right hand grips the top handle of a large, rigid structured handbag in matching warm beige — clean corners, a minimal clasp hardware detail, smooth leather surface with a faint specular highlight along the top edge. Her left arm swings naturally forward in walking rhythm. The setting is a quiet urban street with an unbroken flat gray concrete wall running parallel behind her — its surface showing fine aggregate texture, faint weathering marks, and a long directional shadow cast obliquely across it from soft overhead daylight. The sidewalk beneath is smooth pale concrete, a subtle extension of the muted pastel palette. The entire environment stays restrained — no signage, no distracting props — allowing the subject and garment textures to own the frame completely. Lighting is soft diffused daylight from a high overcast sky, approximately 5500K, producing gentle directional shadows that fall with enough depth to sculpt the knit fabric folds and give the suede boots dimensional presence without blowing any highlights. A faint secondary bounce from the pale concrete wall provides a cool fill from the right, keeping shadow areas luminous rather than dead. Shot on a Sony A7R V with a 85mm f/1.8 lens, three-quarter framing from just below the boot heel to just above the crown — subject filling the vertical frame with breathing room on both sides. Subtle cinematic depth of field with the subject in full sharp focus, the concrete wall behind her transitioning to a very slight soft focus toward the frame edges. Individual knit fiber strands catching sidelight, suede nap microscopically rendered, natural hair movement from stride, fine skin texture on the exposed shoulder and collarbone. High-fashion street photography editorial style. Muted, sophisticated color palette anchored by warm beige and brown against cool concrete gray. Vibrant but restrained — no oversaturation. 8K resolution, high dynamic range, no watermarks, no text in frame, no motion blur on subject, clean wall background with no graffiti.
```

Use `seed=42` (matches `handler.py`'s own default, so omitting `seed` from the
job input is equivalent) and leave every other parameter at its default for
both the "before" and "after" calls — only the code changes should differ
between the two runs. Include `TEST_LORA_URL` in `loras` at its default scale
(`0.85`) for both calls so multi-LoRA quality is part of what's being judged.

## Pre-authorized actions (for non-interactive `/goal` runs)

The user has explicitly pre-authorized, for this goal's scope only, the loop
performing these two actions **without pausing to ask**:

1. `git push` (to the branch this work happens on) to trigger RunPod's
   auto-build after implementing the quality changes.
2. The `PATCH https://rest.runpod.io/v1/endpoints/{endpointId}` rolling-
   release call against the **live production endpoint**, as described in
   the Best-effort bonus section below.

This authorization is scoped to exactly these two actions for this goal; it
is not a blanket authorization for other risky operations (e.g. force-push,
deleting endpoints/templates, changing `workersMin`/`workersMax`, modifying
other production infrastructure) — those still require asking first.

## Best-effort bonus (attempt, document, but never blocks success)

- Load `RUNPOD_ENDPOINT_URL`, `RUNPOD_API_KEY`, and `TEST_LORA_URL` from
  `.env` (e.g. via `python-dotenv` or a simple parser — don't add a new
  runtime dependency to the Dockerfile for this, it's a local dev/test-only
  concern). If any value is still the literal placeholder `REPLACE_ME` or
  the file/values are missing, ask the user once (a single
  `AskUserQuestion`) whether they want to supply real values now; if they
  decline or none are available, explicitly note in the report that live
  validation was skipped — never let this block or stall the loop.
- Confirm the request/response shape for `RUNPOD_ENDPOINT_URL` (typically
  `/runsync`, blocking, `{"input": {...}}` in and `{"output": {...}}` out,
  `Authorization: Bearer <RUNPOD_API_KEY>` header) against a real call before
  relying on it — don't assume the shape from memory or from this doc alone.
- **Before** making any quality changes: call `RUNPOD_ENDPOINT_URL` with the
  exact prompt from **Fixed test prompt** above, `seed=42`, and
  `TEST_LORA_URL` in `loras`, save the returned image URL as the "before"
  sample.
- Implement the code changes, commit, and `git push` to trigger RunPod's
  auto-build.
- Once the new image has built (poll RunPod's build/deploy status if an
  endpoint for that exists; otherwise wait a reasonable fixed interval and
  note the assumption in the report), force the endpoint onto the new image
  by calling `PATCH https://rest.runpod.io/v1/endpoints/{endpointId}` with
  `Authorization: Bearer <RUNPOD_API_KEY>` (an empty/no-op JSON body is
  sufficient — the PATCH call itself triggers RunPod's rolling release, per
  https://docs.runpod.io/api-reference/endpoints/PATCH/endpoints/endpointId).
  `RUNPOD_ENDPOINT_URL` in `.env` gives the run/runsync host+endpointId; the
  management-API `endpointId` path segment is the same ID, just called
  against `rest.runpod.io/v1` instead of `api.runpod.ai/v2`.
- **After** the rolling release completes: repeat the exact same "before" job
  input against `RUNPOD_ENDPOINT_URL`, save the "after" image URL. If the
  rolling release could not be confirmed to have picked up the new image
  (e.g. no build-status API available), say so explicitly in the report
  rather than presenting the comparison as conclusive.
- Include both image URLs and a written visual comparison (the agent can
  view both images) in the final report. Save both URLs, the exact job input
  used, and the redeploy call's response to `.goals/decision-log.md` so the
  comparison is reproducible.

## Explicitly out of scope / don't touch

- `s3_utils.py` — S3 upload plumbing, unrelated to image quality.
- `.claude/`, `.serena/`, `.codegraph/`, `AGENTS.md`, `.mcp.json` — tooling
  config, not part of the worker itself.
- Reintroducing `requirements.txt` as a real dependency source — it is
  intentionally unused (deps live in `Dockerfile`/`runpod_bootstrap.sh` per
  commit `4f5def1`); don't change that convention as a side effect.

## Turn cap and failure path

**20 tries.** Reasoning: this spans multiple real phases — firecrawl web
research, context7 documentation lookups, a full code-review pass,
multi-parameter implementation changes in an ~800-line handler, setting up a
lint config from scratch and iterating it to clean, and a line-by-line
README/code cross-check — each of which can take a few turns on its own, but
none of which is open-ended file-by-file refactoring across the repo.

**If still failing after 20 tries:** stop. Revert `handler.py`, `Dockerfile`,
`runpod_bootstrap.sh`, `requirements.txt`, `README.md`, and `.flake8` to
`HEAD` (`git checkout -- <files>`). **Keep** `.goals/decision-log.md` and any
other files under `.goals/` — the research is valuable even if the
implementation didn't converge. Report exactly which success criterion is
still failing and what was tried against it.

## The /goal invocation

```
/goal All of the following hold: (1) `flake8 .` exits 0 using the project's
.flake8 config; (2) every image-quality-affecting parameter change in
handler.py has a cited entry (URL + justification) in
.goals/decision-log.md — official Z-Image/Tongyi-MAI source, HF/diffusers
docs via context7, or a firecrawl-verified source, never assumption-only;
(3) a code-review pass on the diff shows zero unresolved CONFIRMED
correctness findings; (4) every job_input.get(...)/os.environ.get(...)
default in handler.py is accurately documented in README.md with matching
default values (verified by a shown cross-check, zero mismatches); (5)
`python3 -m py_compile handler.py` exits 0; (6) `git diff --stat` touches
only handler.py, Dockerfile, runpod_bootstrap.sh, requirements.txt,
README.md, .flake8, and .goals/*. Best-effort, non-blocking: load
RUNPOD_ENDPOINT_URL/RUNPOD_API_KEY/TEST_LORA_URL from .env (ask the user
once via AskUserQuestion if any is still the REPLACE_ME placeholder or
missing); call the live endpoint with the exact prompt from the "Fixed test
prompt" section of this file, seed=42, and TEST_LORA_URL in loras, as the
"before" sample; implement the changes, then `git push` (pre-authorized by
the user for this goal, no need to pause and ask) to trigger RunPod's
auto-build; once built, force the endpoint onto the new image via `PATCH
https://rest.runpod.io/v1/endpoints/{endpointId}` (Authorization: Bearer
RUNPOD_API_KEY, no-op body — this triggers RunPod's documented rolling
release, confirmed via
https://docs.runpod.io/api-reference/endpoints/PATCH/endpoints/endpointId;
this call against the live production endpoint is also pre-authorized by
the user for this goal, no need to pause and ask — see "Pre-authorized
actions" section of this file for the exact scope), then repeat the same
job input as the "after" sample; save both image URLs, the job input, and
the redeploy call's response to .goals/decision-log.md and include a
written visual comparison in the final report. Never let missing
credentials or a failed live call block the required criteria above.
Scope: handler.py, Dockerfile,
runpod_bootstrap.sh, requirements.txt, README.md, .flake8, .goals/*. Don't
touch: s3_utils.py, .claude/, .serena/, .codegraph/, AGENTS.md, .mcp.json.
Stop after 20 tries. If still failing, stop, revert handler.py, Dockerfile,
runpod_bootstrap.sh, requirements.txt, README.md, and .flake8 to HEAD (keep
.goals/), and report which check is still failing and what was tried.
```
