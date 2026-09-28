# SILOTIGER-1047 — Code review fixes

Close the items in `general-review.md` (2026-09-24, review HEAD `b8ee4802e`)
that a 2026-09-28 source triage still found in the tree. Kernels have since
moved to `aiter/ops/flydsl/kernels/qsa/k1.py` and `k2.py`; the review's
`k1_family_a.py` / `k2_family_a.py` line numbers are historical. Follow the
current files.

Parent: [SILOTIGER-1040](https://amd.atlassian.net/browse/SILOTIGER-1040).
Ticket: [SILOTIGER-1047](https://amd.atlassian.net/browse/SILOTIGER-1047).
Ticket-wide plan: `SILOTIGER-1047-plan.md`. Review: `general-review.md`.

**In scope:** the still-open correctness, safety, test, cleanup, and AOT
items below. Each kernel change is re-timed on the bar shapes and reverted
if it slows them down.

**Out of scope (stale, or already closed on the parent plan):**

- Family A/B emit-kernel and wrapper duplication, and the dead `_VEC`.
  One `k1.py`, one emit module, one `qsa_k1_block_ids`. `H=8` is a second
  scorer compile.
- The unused vector the review cited at old `k2_family_a.py:630-636`. It
  is gone. Dead tiled-copy values that remain are phase 5.
- Competitor pin and the layer profiler. Parent phase 1 is checked.
- Production opt-in. Parent phase 4 is checked: `qsa_layer` is the in-tree
  `auto` / `flydsl` / `triton` entry. `qwen4_exp` is not in this tree, and
  the parent plan says that is intentional. Do not reopen it here.
- K1 long-row tie, sentinel, and expansion checks that already passed, and
  the overlay barriers the review found adequate. Do not rework them.

Work the phases **in order**. Later items assume earlier ones have landed.
Leave checkboxes unchecked until that item is done; paste tables and notes
under the relevant phase as evidence.

## Progress

- [ ] 1. K2 correctness (4 GiB pages, empty-tile NaN, output layout, Q over-read, empty tables)
- [ ] 2. K1 page faults (prefill over-read, decode page ids, K1 4 GiB / i32)
- [ ] 3. K1 gfx942 H=8 LDS and the runtime arch allowlist
- [ ] 4. Score-matrix scope, benches, and stronger K1 assertions
- [ ] 5. Behavior-preserving kernel cleanup
- [ ] 6. QSA AOT registration
- [ ] 7. Optional: caller-owned K2 split workspace, only if a profile says so

## Locked decisions

These locks apply to **this campaign** unless a later note explicitly
supersedes them. Ticket-wide locks in `SILOTIGER-1047-plan.md` still
apply. This campaign does not reopen that plan's closed phases.

### Skills

- **Skills (read, do not recall).** Before writing or reviewing FlyDSL
  for this ticket, Read
  `.claude/skills/flydsl-kernel-authoring/SKILL.md` and follow it.
  For `op_tests/test_flydsl_qsa.py`, also Read
  `.claude/skills/aiter-op-test/SKILL.md`.
  Do not start kernel code from memory of those skills.
  Cleanup/modernization of existing kernels uses
  `flydsl-kernel-code-cleanup`, not authoring, and is not a phase
  of this plan unless a later lock says otherwise.

Phase 5 is that later lock, for this campaign only. It is a
behavior-preserving cleanup. It does not authorize a kernel rewrite.

### Test environment

- **Test environment:** run all tests/benches in **`flydsl_venv`** **GPU 6**
  (`HIP_VISIBLE_DEVICES=6`).
- **Compile cache.** After kernel-source edits, run with
  `FLYDSL_RUNTIME_ENABLE_CACHE=0` (or clear `~/.flydsl/cache`) so a stale
  HSACO cannot mask a bad rewrite. `CACHE=0` in this container also needs
  `ROCM_PATH=/root/.flydsl/toolkit`, or `gpu-module-to-binary` fails to
  find `ld.lld`. `CACHE=1` hides that by hitting the disk cache.
- **Gate.** Both layers, after each kernel or test edit:
  ```bash
  source /path/to/flydsl_venv/bin/activate
  HIP_VISIBLE_DEVICES=6 FLYDSL_RUNTIME_ENABLE_CACHE=0 \
    ROCM_PATH=/root/.flydsl/toolkit \
    python3 -m pytest op_tests/test_flydsl_qsa.py -q
  ```
  Paths may move; keep this snippet in sync. The full `__main__` sweep is
  the perf check below, not a prerequisite for a correctness-only checkbox.
  New cases are `test_*` in `op_tests/test_flydsl_qsa.py`. Do not put them
  on a `bench_*`.

### Performance

- **Fixes must not hurt performance.** A correctness or safety change that
  slows the serving path is a miss and reverts, even if the numerics are
  right. Flat within run-to-run noise is success. This campaign does not
  use the overlay plan's 3% keep-gate; that gate is for optimizations that
  have to win. Here a speedup is not required, and a loss is not allowed.
- **Bar shapes.** Re-time the kernel you touched, family A, GPU 6 / gfx950,
  cold `--rotate 0`, `FLYDSL_RUNTIME_ENABLE_CACHE=0`:
  - K2: decode `M∈{1,8}` at `L=32768`, prefill `M=512` at `L=8192`
  - K1: the same points, plus one long-row prefill (`M>=16`, one request,
    `n_columns>512`) so the prefill scorer is the thing that ran
  Paste the before/after µs under the phase. Compare like with like: same
  rotate, same cache setting, HEAD control taken with `CACHE=0` immediately
  before the candidate. Do not compare a cold cell to a hot one.
- **Cheap form of each fix.** The shape of the fix is part of the lock.
  A slower implementation that also fixes the bug does not count.
  - **4 GiB pages.** Rebase the page base in 64-bit arithmetic once per
    page, then use a bounded page-local descriptor whose offset stays
    32-bit. Do not turn the K/V gather into a 64-bit flat address per
    element. `max_size=False` on the whole cache is not the fix.
  - **Empty-tile NaN.** When the previous max is `-inf` or the previous
    denominator is 0, rescale by 0. One predicate on the existing
    `alpha`. Do not add a second softmax, a branch per element, or a
    change to the finite-score path.
  - **Output layout.** Allocate the default `out` with a contiguous
    `torch.empty(q.shape, dtype=q.dtype, device=q.device)`. Do not put
    layout into the plan key: that multiplies compiles and can keep the
    first call's stride forever. Caller-supplied `out` stays
    contiguous-required.
  - **Q over-read.** Clamp dead query rows before the address is formed,
    the way K1 already does, or bound the descriptor to the live group.
    Do not add a per-element address on the Q load.
  - **Empty page table or cache.** Reject the geometry, or take a
    zero-output return, before launch. Do not add a host scan of the
    table or a no-load branch inside the tile loop.
  - **K1 page ids.** Add the same `phys_live` predicate K2 already has,
    and bound prefill `col_live` by the request context, before the
    load. Do not change score math, the selector, or `n_columns`.
    Padded tables may still inflate the score buffer; fixing that
    allocation is not part of the fault fix, because it can change
    which scorer runs.
  - **gfx942 H=8 LDS.** Route that one case to the one-row scorer, or
    shrink only the gfx942 tile. Do not shrink the gfx950 H=8 prefill
    tile and do not change gfx950 codegen to make the gfx942 budget.
  - **Arch allowlist.** Reject an unsupported runtime arch in the QSA
    wrappers, from `q.device`. Do not retune gfx950 or gfx942 codegen.
    Do not change `topk_select`'s global `get_gfx()` gate for other ops;
    the QSA wrapper must not call it on a device it has not allowlisted.
  - **Score matrix.** Do not remove the long-row `[M, n_columns]` fp32
    buffer. The parent plan already allows it. The fix is to make
    `SILOTIGER-1047.md` say the same thing.
  - **Benches and assertions.** Test and harness edits only.
  - **Cleanup.** Delete or correct comments and dead values. No numeric
    and no schedule change. If a "cleanup" moves an instruction, it is
    not cleanup.
  - **AOT.** Register with the existing `default_jobs()` collector.
    Do not add a `[None]` sentinel. Registration must not add work to
    the JIT call path.
  - **Split workspace.** Leave the internal fp32 partials in place
    unless a profile on the bar shapes shows that allocation. A
    caller-owned buffer is an API addition, not a second allocation
    on the default path.

### Scope

- **One defect per experiment.** Land and re-time before starting the
  next kernel edit. A phase checkbox stays open until its new `test_*`
  fails on the unfixed tree (or is a static check that cannot run on
  gfx950) and passes after the fix.
- **gfx950 is the device we have.** GPU 6 proves the gfx950 numerics and
  the bar timings. The gfx942 LDS miss is a static shared-memory size
  plus a dispatch decision. Do not claim a gfx942 run from a gfx950 box.
- **Do not touch the overlay barriers** unless a fix in this plan cannot
  be correct without a new one. The review found no synchronization defect.
- **Do not change K1 tie policy, sentinel padding, or expand.** Those
  checks passed. Set equality stays the K1 gate; phase 4 adds duplicate
  and prefix checks on top of it.

## Subtasks

### 1. K2 correctness

Review must-fix 1, 2, 3, and 5, plus the known Q over-read the review
counted separately. All five are still in `k2.py`.

- [ ] **1a. Page addressing past 4 GiB.** `make_buffer_tensor` on the whole
      `k_cache` / `v_cache` (`k2.py` ~341) then `fx.slice` by physical page
      (~566, ~580) uses a 32-bit descriptor offset. Page size 16, physical
      page 262144 aliases page 0. Rebase in 64-bit once per page, then a
      page-local descriptor. Add a `test_*` that fills page 0 and page
      262144 with different values and checks the gather. A multi-gigabyte
      allocation may be skipped when the device cannot hold it; record
      the skip reason. Re-time K2 on the bar shapes.
- [ ] **1b. Empty first tile.** `m` starts at `-inf` (~557). An all-masked
      tile has `tile_max == -inf`, and `alpha = exp2(m_prev - m_new)` (~839)
      is NaN. The epilogue then writes zero because the denominator is not
      `> 0` (~989). Guard the rescale. `test_*`: Q/K zero, V one, indices
      all `-1` except one valid column inside the split (the review's
      `M=1, W=2048` column 16 and `M=512, W=128` column 64). Output is the
      valid token, not zero. Re-time K2 on the bar shapes.
- [ ] **1c. Output layout vs the plan cache.** Default `out` is
      `torch.empty_like(q)` (~1218), which keeps a non-contiguous layout.
      Only a caller-supplied `out` is checked for contiguity. `_plan`
      (~1145) has no layout, and `_run_compiled` keeps the first compile.
      Allocate the default `out` contiguously. `test_*`: alternating
      `[2, 24, 256]` outputs with strides `(6144, 256, 1)` and
      `(6144, 1, 24)` match the oracle. Re-time K2 on the bar shapes
      (the contiguous path is the one serving uses).
- [ ] **1d. Q over-read.** Q is still a 16-row tile (~516) masked after
      the load. Group 12 and group 5 both read past the live heads.
      Clamp before the address, or bound the descriptor to the live
      group. Existing family A and family B K2 oracle tests stay green.
      Re-time K2 on the bar shapes.
- [ ] **1e. Empty page table or cache.** `qsa_k2_serves` does not reject
      a zero-page cache or a zero-width table. The early return (~1230)
      covers only zero rows or a zero-width index list. Safe indices
      clamp to 0 and the gather still loads. Reject, or return zeros
      before launch, when the cache or the table has no pages and the
      index list is nonempty. `test_*` covers both geometries. No bar
      re-time beyond a smoke call: the hot path is unchanged.

### 2. K1 page faults

The K1 tile-extent audit. The prefill fault was reproduced. The decode
fault, the K1 4 GiB cap, and the i32 score index were not run; verify
them, and fix the ones that are real. Do not change score math.

- [ ] **2a. Prefill page-table over-read.** `qsa_k1_prefill_scores_kernel`
      sets `col_live` from `score_col < n_columns` only (~500), loads
      `page_table[0, logical_page]` unchecked (~504), and applies
      visibility at the store (~553). The K descriptor is default
      `max_size=True` (~425). Bound `col_live` by `context_lens[0] / R`
      and require `phys_live`, with a safe page id, before the load.
      `test_*`: one request, `M=32`, context 4096, 64 real pages, four
      padded entries. Pad 0 matches the oracle. Pad `-1` and a huge
      page id do not fault and do not change the selected set. Re-time
      the long-row K1 prefill bar point.
- [ ] **2b. Decode page ids.** The one-row scorer masks columns (~259)
      but still loads `page_table[safe_req, logical_page]` with no
      `phys_live` (~263). A dead column reads entry 0. A `-1` there, or
      a `-1` inside the visible range, is still a cache index. Same
      predicate as 2a. `test_*` for a no-page request whose entry 0 is
      `-1`, and for a `-1` inside the visible range. Re-time K1 decode
      on the bar shapes.
- [ ] **2c. K1 4 GiB cache.** Same 32-bit descriptor class as 1a, on
      `k_buf` in both scorers (~203, ~425). Apply the 1a page-local
      descriptor. If 1a's helper is shared, use it; do not invent a
      second addressing scheme. Re-time only if the scorer source
      changed.
- [ ] **2d. i32 score-store index.** Not verified. Check whether
      `M * n_columns >= 2^31` (the review's example is `16384 × 131072`)
      overflows the store index. If the serving shapes cannot reach it,
      record that and stop. If they can, fix the index without widening
      the hot score path for the bar shapes. Do not switch the bar-shape
      store to int64 "just in case."

### 3. K1 gfx942 H=8 LDS and arch allowlist

Review must-fix 4 and should-fix 7. Source analysis; GPU 6 is gfx950
and does not prove the LDS budget.

- [ ] **3a. H=8 prefill LDS.** `SharedStorage` for the 16-row scorer
      (`k1.py` ~380) is Q `16·H·128` bf16 + K `32·128` bf16 + 16 ints
      + C `H·2·2·64·4` fp32. At `H=8` that is 73,792 bytes, over the
      65,536-byte gfx942 budget. `H=4` fits. Dispatch is still "one
      request and `M >= 16`" (~642) on every arch. Route gfx942 H=8
      to the one-row scorer, or shrink only that gfx942 tile. gfx950
      H=8 prefill stays on the current builder. Static check: the
      gfx950 H=8 tile size is unchanged. Re-time gfx950 H=8 prefill
      only if that builder's source changed; it should not have.
- [ ] **3b. Runtime arch allowlist.** Both wrappers treat every device
      whose name does not start with `gfx950` as the gfx942 path
      (`k2.py` ~1240, `k1.py` ~638). `topk_select` gates on
      build-environment `get_gfx()`, not `q.device`. Allowlist
      `gfx942` and `gfx950` from `q.device` in the QSA wrappers and
      raise otherwise. Do not change the selector's global gate.
      `test_*`: a mocked or CPU-side check is not required; a direct
      unit of the allowlist helper is enough if it does not need a
      second GPU.

### 4. Score-matrix scope, benches, and K1 assertions

Review should-fix 6, 8, and 9, the "consider" assertion note, and the
still-open ticket-doc half of "Earlier findings." No kernel change.

- [ ] **4a. Resolve the score-matrix conflict in the ticket.** Long rows
      still allocate `[M, n_columns]` fp32 (`k1.py` ~636). The parent
      plan allows that. `SILOTIGER-1047.md` still says K1 writes no full
      score matrix. Update the ticket dump so it matches the parent
      lock: fused emit at `visible <= 512`, materialized scores above
      that. Do not delete the buffer.
- [ ] **4b. K1 bench timings cover the same work.** `bench_qsa_family_a_k1`
      times `qsa_k1_block_ids` and stops at block ids. The vLLM and
      #4882 columns time `qsa_select_paged_tokens` /
      `qsa_4882_select_paged_tokens`, which include expand. Time expand
      on the FlyDSL column too, or stop the competitor columns before
      expand. Say which in the bench docstring. The layer bench is
      already a matched chain; leave it.
- [ ] **4c. Sweep mismatches fail, and requested M is not dropped.**
      Kernel benches return `err` and `__main__` only logs the table.
      FlyDSL K1/K2 loops keep `M <= 8` and `M == 512`, so a requested
      `M` of 64, 2048, or 8192 never appears. Fail the sweep on a
      nonzero K1 set mismatch and on a K2 `err` above the unit-test
      tolerance. Run requested `M` values, or log an explicit skip
      that names the kernel's supported band. Do not silently `continue`.
- [ ] **4d. Stronger K1 assertions.** `_set_mismatch_ratio` compares
      sets and ignores duplicates and order. On the existing K1 unit
      cases, also require no duplicate ids and a compact valid prefix
      followed by `-1`. Set equality stays.

### 5. Behavior-preserving kernel cleanup

Review should-fix 10, plus the audit notes that are dead code rather
than faults. Read `flydsl-kernel-code-cleanup` before editing. No
numeric change, no schedule change, no bar re-time beyond the pytest
gate unless a diff is not comment-only — if the diff is not
comment-only, re-time the touched kernel and revert on a loss.

- [ ] **5a. K2 header.** The module docstring says gfx950 always stores
      K and V separately. Decode still overlays one tile;
      `split_kv_lds` is prefill-only. Make the header match.
- [ ] **5b. Dead K2 values.** `k2.py` ~339 builds a tiled copy and
      discards it. ~370 discards `make_tiled_copy_B(...).get_slice(lane)`.
      Remove them if they have no side effect. Leave `qk_a_copy` and
      `kv_store`; those are used.
- [ ] **5c. Emit's unused arguments.** The emit kernel takes `q`,
      `k_cache`, `page_table`, and `score_scale` and does not read
      them (`k1.py` ~69). Drop them from the kernel if the launch ABI
      can change without a second compiled entry, or stop threading
      them through. Do not change the ids the emit writes.

### 6. QSA AOT registration

"Earlier findings": there is no QSA `OpKind` or collector registration.
`aiter/aot/flydsl/common.py` `OpKind` is still MoE, GEMM, conv, GDN,
and FMHA. The collector does not treat a `[None]` job list as "no
configs"; use `default_jobs()`, the same pattern as `mega_moe.py`.

- [ ] **6a.** Add a QSA `OpKind` and a collector that returns
      `default_jobs()` for the shapes this campaign already compiles
      (family A K1 emit + long-row scorer, family A K2 bar launch).
      Do not register every bench shape. The JIT wrappers stay the
      runtime path; AOT must not add a lookup to `qsa_k1_block_ids`
      or `qsa_k2`.
- [ ] **6b.** A collector unit check: an empty config list is an empty
      job list, not a `[None]` sentinel. No GPU required.

### 7. Optional split workspace

"Consider" in the review. Not required to close the review.

`qsa_k2` allocates fp32 partials on every split launch (`k2.py`
~1249). At family A, `M=128` and 8 splits, that is about 24 MiB. It
was not measured.

- [ ] **7a.** Profile the bar shapes. If the allocation is not visible
      against the kernel, check this box with the profile note and
      do not change the API.
- [ ] **7b.** If it is visible, accept a caller-owned workspace and
      keep the current allocation as the default when the caller
      passes none. Re-time the bar shapes. The default path must not
      get a second buffer.
