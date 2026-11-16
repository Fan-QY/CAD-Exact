**Revised PDF - change log**

We thank all three reviewers for their detailed reviews, and the Area Chair for coordinating the discussion phase. This revision (27 pages total; main text spans pages 1-10 and ends on page 10, within the original 10-page ICLR submission limit - see Page count at end) implements the changes committed in our individual replies. We summarize the three substantive additions first, then group all changes by the review concern they address, followed by a list of internal corrections.

**(i) Order-invariant, tolerance-based literal recovery.** A new Tier-1 metric **PEM-eq$_{10}$** (TVF-gated, order-invariant full-literal recovery within ±10 grid units) is defined in §3.5, formalised in §E.1, and reported for all eight CadQuery configurations × four operations in Table 3. This responds to the order-invariant exact-match variant asked for by **tDVn (W2)** and to **KAf1 (W3, Q1)**'s equivalence question. The diagnostic still requires a one-to-one match of all reference literals within the ±10 grid-unit tolerance and a TVF@5% pass, so geometrically equivalent parametrizations with different literal values can still fail; canonicalization of equivalent parametrizations remains open work.

**(ii) Noise and multi-operation diagnostics.** A 500-sample Gaussian-noise sweep (§F.5, addressing **KAf1 W1**) and a 1,497-sample zero-shot pilot on two-type Boolean-union compositions (§F.6, addressing **KAf1 W2/Q3** and **99VQ**'s scope concern) show that the executability-exactness gap GeomCAD-Eval exposes persists under input perturbations and under two-type compositions.

**(iii) Shape-metric vs parameter-recovery analysis.** A new appendix §F.7 (addressing **99VQ follow-up Q2**) reports Spearman correlations, a conditional pass-rate analysis under tight IoU/CD thresholds, and four per-sample exemplars, quantifying that strict parameter recovery has |ρ| ≤ 0.19 against IoU/CD where defined.

We additionally release the code, the trained CAD-Exact checkpoints, and the four test splits as anonymous previews during the discussion period (see **Reproducibility artefacts** at the end).

---

**Addressing tDVn (W2) and KAf1 (W3, Q1) - order-invariant / equivalence-aware PEM variant**

Reviewer tDVn (W2) suggested an exact-match variant that is invariant to literal ordering, and Reviewer KAf1 (W3, Q1) raised the related equivalence question (different literal sequences can produce the same solid). (tDVn Q1 was about the staged chain's Stage-2 Extrude result and is answered separately in the per-reviewer reply.)

- **§3.5**: strict PEM definition made explicit ("position by position; no reordering is applied"). PEM-eq$_\tau$ is also defined here as a Tier-1 metric (seventh Tier-1 column of Table 3), with its one-sentence definition pointing to the formal equation in Appendix E.1 and the τ-sweep in Appendix F.4. Tier count updated: thirteen main-table metrics (seven Tier-1 + five Tier-2 + one Tier-3).
- **§5.3**: new paragraph introducing PEM-eq$_\tau$, the TVF-gated, order-invariant full-literal recovery diagnostic.
- **Table 3 (main results)**: new Tier-1 column PEM-eq$_{10}$ reported for all eight CadQuery configurations across four operations. DeepCAD (PC2CAD) is n/a since it emits a non-CadQuery DSL.
- **§E.1 (Metric Definitions)**: formal definition of PEM-eq$_\tau$ added (eq. 7).
- **§F.4**: τ ∈ {1, 5, 10} sweep (Table 15). For CAD-Exact (joint), the TVF-gated recovery rate rises from 0.0% at τ=1 to 27.4% at τ=10 on loft and from 3.5% to 21.5% on revolve: a sizeable share of valid loft predictions is within ±10 grid units of every reference literal even when strict PEM is 0.0. This is a tolerance-recovery observation, not a structural claim about parametrization equivalence (**KAf1 W3**).
- **Abstract**: tightened for strict-PEM clarity. The headline statement now reads "...models recover the reference integer-literal sequence exactly (position-wise) for at most 2.7% of executable predictions, and the highest TVF-gated literal recovery rate at τ = 10 grid units is 28.6% across the evaluated configurations", making the strict position-wise match and the Table 3 denominator explicit.

**Addressing KAf1 (W1) - noise robustness**

Reviewer KAf1 (W1) asked for a noise-level sweep on a subset to characterize robustness to input perturbations.

- **§F.5**: zero-shot evaluation of CAD-Exact (joint) on a 500-sample balanced subset under isotropic Gaussian input noise at σ ∈ {0.001, 0.002, 0.005, 0.01, 0.02} ≈ 1-20 grid levels. Validity drops from 94.2% to 51.0%, aggregate TVF@5% from 73.6% to 7.6%, strict PEM from 4/500 to 0/500. The rPEM margin over a shuffled-reference baseline remains positive across the sweep (+26.5 pp on clean inputs, +4.2 pp at σ = 0.02).
- **§6 (Limitations)**: cites the F.5 numbers.

**Addressing KAf1 (W2, Q3) and 99VQ (scope question) - multi-operation scope**

Reviewer KAf1 (W2, Q3) raised multi-operation and compositional generalization; Reviewer 99VQ's scope question asked why the benchmark is limited to single-operation-type programs.

- **§F.6**: zero-shot pilot on 1,497 two-type Boolean-union compositions (Extrude + Revolve / Sweep / Loft). Validity 76.6%, TVF@5% 22.4%, strict PEM 0.0% (order-invariant exact literal-multiset match also 0/1,497 - simple reordering does not close the gap). Because the model was trained only on single-type programs, this is a **zero-shot out-of-distribution** pilot: it shows that the gap GeomCAD-Eval measures is not removed by composition, not a measurement of the composed task's intrinsic difficulty.
- **§6 (Scope and future work)**: cites the F.6 numbers.

**Addressing 99VQ (Q1 follow-up) - operation type and construction steps**

In a follow-up comment, Reviewer 99VQ asked for the distribution of operation counts across shapes within the entire dataset and within each of the four subsets.

- **§3**: new paragraph "Operation type and construction steps" clarifying that each GeomCAD program uses exactly one generative operation **type** (extrude/revolve/sweep/loft). In the standard release split, 61.2% of Extrude programs contain at least two solid-generation calls (`.box`, `.cylinder`, `.extrude`); Extrude programs may also combine generated solids through Boolean operations. Revolve, Sweep, and Loft programs each use one solid-generation call per program with additional sketch / profile / path construction steps. All construction-step literals are part of the PEM target.
- **§2 (Related Work)**: the "multi-op coverage" phrase in the three-axis summary of GeomCAD's position is replaced with "four-operation-type coverage", consistent with the §3 clarification and with our 99VQ Q1 reply.
- **Follow-up to 99VQ Q1 - `.union()` call rate.** Appendix C.5 reports the proportion of Extrude programs containing `.union()` (61.2% of the pool, 61.6% of the test split). This complements the solid-generation call counts reported in our Q1 reply.

**Addressing 99VQ (Q2 follow-up) - PEM vs shape-metric relationship**

In a follow-up comment, Reviewer 99VQ asked for a correlation analysis between GeomCAD-Eval metrics and IoU/CD to justify reporting parameter-recovery metrics alongside shape metrics.

- **§F.7 (new)** carries the analysis committed in our Q2 reply into the appendix:
  - Spearman correlations (Table 19): strict PEM has |ρ| ≤ 0.19 against IoU/CD where defined; CD vs. rPEM shows moderate correlation on Revolve/Sweep/Loft (|ρ| = 0.41-0.62). The two shape metrics agree strongly with each other (CD vs. F@2%: |ρ| = 0.66-0.94), consistent with both measuring geometric similarity.
  - Conditional pass rate (Table 20): restricting to predictions with IoU ≥ 0.95 and CD% ≤ 0.5, the strict-PEM pass rate is still only 0.0-6.0% per operation.
  - Per-sample exemplars (Table 21): one sample per operation with IoU ≥ 0.95, CD% ≤ 0.5, TVF@5% Pass, yet strict PEM = 0 and rPEM = 0.067-0.133.

**Addressing tDVn (W1) - staged vs. joint trade-off in §5.2**

Reviewer tDVn (W1) noted that no single CAD-Exact variant leads across all operations and metrics.

- **§5.2 (main text, Multi-Operation Comparison)**: this section now states the trade-off directly: "on extrude the joint model is within 1.0 pp of the best staged checkpoint on TVF@5% (84.3% vs. 85.3%); it trails on revolve (TVF@5% 77.7 vs. 82.9) and sweep (TVF@5% 49.0 vs. 59.2), and leads by a wide margin on loft (TVF@5% 94.8 vs. 58.4, IoU 96.5 vs. 84.3, Val 96.6 vs. 61.9)", alongside the existing analysis in §6 (Limitations).

**Addressing 99VQ (W3) - baseline reproducibility**

Reviewer 99VQ (W3) noted the limited number of baselines in Table 3.

- **§5.4 (External Evaluation)**: added that CAD-SIGNet and TransCAD do not release public inference code; numbers for both on the DeepCAD split are taken from their respective papers, consistent with the "coarse positioning rather than head-to-head" caveat already present.

**Addressing 99VQ (W4) - DeepCAD leaderboard (Table 13)**

Reviewer 99VQ (W4) noted CAD-Exact's substantially worse (higher) median Chamfer distance on the DeepCAD leaderboard (Table 13).

- **§5.4 (External Evaluation)** now spells out both DeepCAD outcomes in the main text: under cadrille's best-of-10 protocol, zero-shot CAD-Exact (joint) ranks **5th of 12 on median Chamfer distance** (2.46) and **1st of 12 on invalidity** (0.25%) - the invalidity result complements rather than cancels the CD gap. Under best-of-1 (matching Table 3), executability is 75.2% / 65.0% / 56.5% for joint / stage 1 / stage 4. CAD-Exact is evaluated zero-shot on DeepCAD and is trained on GeomCAD (capacity split across four operation types); the leading methods by median CD on this split are trained on DeepCAD itself or on its parent CAD-Recode 1M corpus with the same Onshape sketch-and-extrude provenance.

**Addressing tDVn (Q2) - unresolved cross-references**

- The two cross-references flagged in the submitted PDF (near §5 and §F) are now resolved in the revised build.

---

**Internal corrections** (not raised by reviewers, discovered during the data audit)

**Impact on reported numbers.** No Table 3 result changes as a consequence of these corrections, except the stage-3 Sweep PEM-eq$_{10}$ rounding fix (22.3% -> 22.4%); the added PEM-eq$_{10}$ column is a new metric, not a correction. The original strict-PEM headline of 2.7% in the abstract is unchanged; the revised abstract adds the 28.6% PEM-eq$_{10}$ result as a new reported quantity.

- **Corpus count**: 4,006,182 -> **4,006,426**. The 244-sample difference is the **244 additional loft test programs** actually used to score the loft column of Table 3, which were omitted from the earlier count. The released split (standardized to 1,500 test samples per operation) is unaffected; the extra 244 loft test programs ship as a supplementary file alongside the release so the paper-version loft column (test = 1,744) is reproducible. The corpus distribution table in our 99VQ Q1 reply counts the released split and is otherwise unaffected.

- **§C Table 7 (per-operation stats)**: row order aligned to Extrude / Revolve / Sweep / Loft, matching Tables 3 and 4.

- **§D LoRA parameterisation**: eq. (1) (appendix D) updated from W' = W + BA to **W' = W + (α/r)·BA** with α = 128 explicitly stated (α/r = 2); staged-arm merge formula updated correspondingly.

- **§D Optimization paragraph**: corrected the hardware/batch decomposition to match the released trainer-state records - the joint arm used per-device batch 32 × gradient accumulation 1 on 2 GPUs, and each staged stage used per-device batch 16 × gradient accumulation 4 on 1 GPU (both yielding effective batch 64, as previously stated). The hardware line was updated from "4 × A100 80 GB" to "NVIDIA H100 and H200 80-141 GB GPUs", which matches the actual runs.

- **§B validity-gate narrative** (correction to a data-quality claim): the earlier submission stated that watertightness was enforced at generation time; **in fact watertightness is audited post hoc and flagged per sample (Appendix C.6)**, while the generation-time validity gate is execution plus positive volume (shared) and a `trimesh` read-back (composed pipelines only). Audit result on 200 samples per operation: 91.0% extrude, 99.0% revolve, 85.0% sweep, 100.0% loft (Appendix C.6). The algorithm pseudocode has been updated accordingly.

- **§B pseudocode (Algorithms 1–3)** (the following corrections to match the released code):
  - Alg. 1 line 6 (Extrude auto-translate): per-axis conditional and max(1, ·) clamp added.
  - Alg. 3 line 5 (degenerate range): literals mapped to a constant in [t_min, t_max] rather than rejected.
  - Alg. 3 lines 1 and 10 (sentinel Z): explicitly restricted to the composed pipelines (loft/revolve/sweep); the extrude in-place rewrite does not apply the mask.
  - Alg. 1 line 50 and Alg. 2 line 71 (STL export + validity check): corrected to use CadQuery's `cq.exporters.export`; the read-back check uses `trimesh.load` with a non-empty-triangle-mesh requirement (not watertightness or a specific minimum face count).

- **Table 3 stage-3 Sweep PEM-eq$_{10}$**: rounding corrected (22.35% -> **22.4%**; previously truncated to 22.3%).

- Minor TODO comments removed from the abstract and §F.1.

- **§6 Conclusion and Limitations reorganization**: split into **Conclusion / Limitations / Scope and future work**. The Conclusion states the measurement contribution - the benchmark and scoring protocol turn literal-level parameter recovery into a quantitative target - alongside the CAD-Exact (joint) results (leading every evaluated baseline on every operation in rPEM; matching or leading them on TVF@5% (on extrude the joint model is 0.6 pp ahead of CAD-Recode at 84.3% vs. 83.7%); the best per-operation configuration matches or exceeds every baseline on strict PEM and PEM-eq10), while noting that exact recovery remains open (at most 2.7% of executable predictions). Limitations retains the core caveats (four generative operations; single operation-type scope with clean procedural point clouds; the staged-vs-joint regime-dependent trade-off; the F.5 noise sweep; the DeepCAD/PC2CAD caveat). Scope and future work cites the F.6 two-type composition pilot. No Table 3 number is changed beyond the new PEM-eq10 column.

- **Reproducibility statement**: added a sentence clarifying that the public release standardizes test to 1,500 samples per operation while the 244 additional loft test programs used by Table 3 ship as a supplementary file so the paper-version loft column is reproducible. Also added the Zenodo anonymous-preview link for the released checkpoints (see "Reproducibility artefacts" section below).

---

**Reproducibility artefacts (anonymous preview, available during the discussion period)**

- **Code:** <https://anonymous.4open.science/r/CAD-Exact-5707> (updated 2026-11-16)
- **Weights and test splits:** <https://doi.org/10.5281/zenodo.24158668> (CC-BY-NC-4.0; concept DOI <https://doi.org/10.5281/zenodo.24158667>)
  - `cad-exact-joint.tar` -> `cad-exact-joint/checkpoint-187500-fused`: reproduces the joint row of Table 3 (all four operation columns, including PEM-eq$_{10}$, and the Appendix F.5 / F.6 / F.7 analyses).
  - `cad-exact-loft.tar` -> `cad-exact-loft/checkpoint-46875-fused`: reproduces the stage-4 (loft) row of Table 3.
  - `GeomCAD_test_splits_v1.tar.gz` (502 MB): the 6,244 test samples scored in Table 3 (1,500 extrude + 1,500 revolve + 1,500 sweep + 1,744 loft). Each sample is `<id>.py` plus `<id>.stl`; the layout matches the `--datainfo` and `--gt_dir` conventions of `infer_batched.py` and `eval.py`. With the joint and stage-4 checkpoints above, this is sufficient to reproduce the joint and stage-4 (loft) rows of Table 3 end to end.
  - The anonymous preview does not include the stage-3 checkpoint associated with the revolve results of 2.7% strict PEM and 28.6% PEM-eq$_{10}$, nor the stage-1 and stage-2 checkpoints; those, together with the full 4M-program corpus and the generator and training code, will be released upon acceptance.

---

**Page count**: 27 pages overall. The main text (Sections 1-6) ends on page 10, within the original 10-page ICLR submission limit - we did not use the +1-page discussion-period allowance. The AI Use, Ethics, and Reproducibility statements follow on page 11 onward; these three statements, together with the references and the appendix (ending at page 27, one page longer than the submitted version due to the new §F.7 subsection), are not counted against the main-text limit per ICLR policy. All Table 3 numbers other than the new PEM-eq10 column are unchanged from the submitted version; the abstract was updated for strict-PEM clarity as noted above, and its final sentence now also lists the test splits as part of the review-time anonymous previews (previously only code and the joint and stage-4 checkpoints were listed).

---

We hope these additions address the concerns raised and would be grateful if the reviewers could reconsider their ratings in light of the revised PDF, the per-reviewer discussion threads, and the reproducibility artefacts linked above.
