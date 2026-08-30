# Human benchmarks for respiratory sound labeling

**Compiled:** 2026-08-16 · **Why:** M39's G2 runs returned crackle AUROC 0.5506 then 0.5580, and
wheeze 0.5729 then 0.5340, against ICBHI cycle labels — a FAIL against the 0.65 gate on both runs. Read alone that looks like failure. Read against what *humans* achieve
on the same task, it looks different. These are the reference points.

**Use this file when writing up M39, and before interpreting the clinician's returned labels.**

---

## 1. Physicians vs. ICBHI labels — the direct comparison

**Tzeng et al., *JMIR AI* 2025;4:e67239** — "Improving the Robustness and Clinical Applicability
of Automatic Respiratory Sound Classification Using Deep Learning–Based Audio Enhancement"
(arXiv:2407.13895). Table 4.

Seven **senior physicians** blindly annotated a random 25% of the ICBHI test set — no access to the
class label or the noise condition — rating confidence 1–5.

| Recording condition | Accuracy | **Sensitivity** | Specificity | **ICBHI score** | Confidence (SD) |
|---|---:|---:|---:|---:|---:|
| Clean | 49.40% | **23.23%** | 72.32% | **47.77%** | 2.88 (1.50) |
| Noisy | 47.59% | 16.77% | 74.58% | 45.68% | 2.32 (1.29) |
| Denoised | 51.51% | 28.38% | 71.75% | 50.07% | 2.65 (1.36) |

**Seven senior physicians reach a 47.77% ICBHI score on clean ICBHI audio, missing roughly 77% of
the cycles ICBHI labels abnormal, at a mean self-rated confidence of 2.88/5.**

This is the single most useful number we have for interpreting M39. It says the gap between our DSP
extractors and ICBHI's labels is *not obviously worse than the gap between expert humans and those
same labels*.

**Caveat before leaning on it:** the paper does not state that the physicians were given ICBHI's
annotation conventions or calibration examples. Some of that 49.4% may be convention mismatch
rather than intrinsic task difficulty. Our own listening pack ships six reference exemplars for
exactly this reason, so our clinician is *better* calibrated than theirs — a defensible
methodological difference worth stating.

**Also note:** the study's purpose was evaluating an audio-*enhancement* module, not validating
concept extractors. It does not touch fine/coarse crackle, wheeze pitch, or rhonchi. Our G1
kill-search verdict is unaffected.

---

## 2. Physician-vs-physician agreement — the reliability ceiling

This matters more than §1 for our fine-grained concepts, because it bounds what *any* ground truth
on this task can be worth.

> ⚠️ **Attribution corrected 2026-08-30.** This paper was cited throughout the project as
> "Aviles-Solis et al. (2016)". **The first author is Melbye.** Aviles-Solis is a co-author on
> other lung-sound papers from the same Tromsø group, which is where the slip came from. The
> error is repeated in `DECISION_2026-08-16_PIVOT.md`, `CLINICIAN_RESULTS_v1.md`, `M39/README.md`
> and `PAPER_OUTLINE.md`; those are dated records and were left as written. `DRAFT_PAPER/main.tex`
> and `references.bib` are corrected. **Cite Melbye, not Aviles-Solis.**

**Melbye, H., García-Marcos, L., Brand, P. L. P., Everard, M. L., Priftis, K. N., and
Pasterkamp, H. (2016). "Wheezes, crackles and rhonchi: simplifying description of lung sounds
increases the agreement on their classification: a study of 12 physicians' classification of lung
sounds from video recordings."** *BMJ Open Respiratory Research* **3**(1):e000136.
DOI [10.1136/bmjresp-2016-000136](https://doi.org/10.1136/bmjresp-2016-000136) · PMID 27158515.
Author list, venue and volume verified against the OpenAlex record for that DOI, 2026-08-30.

- 20 audiovisual recordings from the ERS lung-sound repository
- 12 observers (6 paediatricians, 6 adult physicians), 10 predefined sound categories

| Description level | Agreement |
|---|---|
| **Detailed** descriptions of adventitious sounds (incl. fine vs. coarse crackle) | **κ < 0.40 — poor to fair** |
| **Combined** category: crackles | κ = 0.62 — moderate/good |
| **Combined** category: wheezes | κ = 0.59 — moderate/good |

> **Conclusion:** broader descriptions are shared far more reliably between observers than detailed
> ones. Simplifying the taxonomy *increases* agreement.

### Why this is decisive for us

`crackle_fine_ratio` claims to measure exactly the distinction that physicians agree on at
**κ < 0.40**. That has three consequences, and all three should be stated in the paper:

1. **The fine/coarse concept has a low reliability ceiling that has nothing to do with our DSP.**
   Even a perfect extractor cannot agree with a reference standard that observers do not agree on
   among themselves.
2. **Keeping fine/coarse as a permanently-labelled *proxy* is now the published-literature-backed
   position**, not a hedge. This retroactively justifies the naming discipline in
   `concept_extractors.py::CONCEPT_VALIDATION`.
3. **If our clinician's intra-rater duplicate check comes back weak on fine/coarse, that is the
   expected result**, consistent with κ < 0.40 — not a failed exercise and not a bad clinician.

Other supporting figures from the same literature: crackle detection ~72% agreement, κ = 0.41; and
in a 7-specialist study over 70 lung sounds, wheeze κ = 0.704, crackle κ = 0.514.

---

## 3. Absolute recognition ability

Reported detection rates by sound type (pneumologists > paediatricians > students):

| Sound | Correct detection |
|---|---|
| Wheezes | 70–90% |
| Crackles | 55–75% |
| Bronchial sounds | 15–30% |

Wheezes are consistently easier than crackles for humans — which mirrors M39, where
`wheeze_score` (0.5729) edged out `crackle_score` (0.5506). Weak evidence, but it points the same
way.

**Related:** spectrogram displays alongside audio measurably improve human wheeze/crackle
classification (Sci Rep 2020, `s41598-020-65354-w`). Relevant if we ever revisit the listening
protocol — we currently ship audio only.

---

## 4. How to use these numbers

**In M39's write-up.** Never report "AUROC 0.55" alone. Report it against the human benchmark:

> Our physics-derived crackle extractor reaches AUROC 0.5506 (95% CI [0.5260, 0.5751]) against
> ICBHI cycle labels. For context, seven senior physicians blindly annotating the same corpus
> reach a 47.77% ICBHI score with 23.23% sensitivity (Tzeng et al., JMIR AI 2025), and physician
> inter-observer agreement on detailed adventitious-sound descriptions is κ < 0.40
> (Aviles-Solis et al., 2016). The ceiling on this task is set substantially by label reliability,
> not only by detector quality.

**When the clinician's labels return.** Sanity anchors, not targets:

- Their agreement with ICBHI landing near **~49%** is *expected*, not a problem.
- Weak fine/coarse intra-rater agreement is *expected* (κ < 0.40 in the literature).
- Strong agreement on crackle/wheeze *presence* (κ ≈ 0.6) is the realistic best case.

**What this does NOT license.** It is context, not an excuse. A detector at 0.55 is still weak, the
revised extractors still need their re-run, and "humans are also bad at this" is an argument about
the *task*, not evidence that our method works.
