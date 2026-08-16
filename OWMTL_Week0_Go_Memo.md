/ OWMTL — Week 0 Go Memo

**Prepared:** 2026-08-15 · **Covers:** G0 (annotation), G1 (kill-search), G4/CC2 (device structure), protocol freeze
**Status:** ✅ G1 clear · ✅ G4 answered · ✅ CC1 done · ⚠️ G0 awaiting clinician answer

---

## G4 / CC2 — Device structure ✅ ANSWERED: **drop the device axis**

Run from the committed official split file (device is encoded in every ICBHI filename, so this
needed no audio download).

| | |
|---|---|
| **Devices** | **4** — AKGC417L 646 (70.2%), Meditron 128 (13.9%), LittC2SE 86 (9.3%), Litt3200 60 (6.5%) |
| **Chest locations** | 7 — Al, Ar, Ll, Lr, Pl, Pr, Tc |
| **Patients spanning >1 device** | **3 of 126 (2.4%)** |

**Verdict: BLOCKING — device is effectively a patient-level property.** 123 of 126 patients were
recorded on exactly one device, so leave-one-device-out *is* leave-those-patients-out. Any measured
"device effect" would be inseparable from inter-patient variation, and a within-patient device
contrast has n=3.

**Per roadmap G4 "if not satisfied": drop the device axis, rely on pediatric (SPRSound) shift as the
covariate stress, and say so explicitly in the paper.** This is exactly the cheap insurance CC2 was
meant to buy — the experiment is impossible, and we now know before building it rather than after.

> Also settles a factual error that had propagated into planning: ICBHI has **4** recording devices,
> not 7. The 7 is the chest-location count. `Research_Progress_Report.md` has been corrected.

---

## G0 — Annotation decision ⚠️ NEEDS DR. KHAN

**The question:** hand-label ~150–300 ICBHI cycles with fine-grained acoustic concepts
(fine/coarse crackle, inspiratory phase, wheeze pitch) against a clinical reference?

### The one fact that decides it

**Is there a clinician who will validate the labels?** Nothing in the repo identifies one. The
roadmap itself lists "access to a clinician/reference for label validation" under *evidence needed*
— and that line is still blank.

This is decisive because **fine vs. coarse crackle is a genuine clinical discrimination**, not a
transcription task. Three CSE students self-labeling produces an asset of *unknown validity*.
Releasing that as "the first fine-grained concept annotations for ICBHI" would be an unvalidated
claim presented as a contribution — the precise failure mode this project has already been burned by
seven times (`Asif's/CLAUDE.md`, "Traps"). It would also sit badly against the current pitch, which
is *honest evaluation*.

### What "No" actually costs — state this plainly to Dr. Khan

**It does not block the make-or-break gate.** G2 validates the extractors against ICBHI's *existing*
coarse crackle/wheeze labels, which we already have for all 6,898 cycles. The engine build is
unaffected.

**But it does leave the fine-grained concepts unfalsifiable.** Fine/coarse crackle, wheeze pitch
band, and inspiratory phase would be *computed* but never *validated*. A reviewer asks "how do you
know your fine-crackle detector detects fine crackles?" and label-free has no ground-truth answer.
That is a real, foreseeable weakness — not a reason to annotate badly, but a reason to decide
deliberately rather than by default.

### Mitigations if the answer is No

1. Validate the **coarse** concepts (crackle/wheeze presence) against ICBHI labels — fully defensible.
2. For fine-grained concepts, report agreement against the **published detectors they derive from**
   (e.g. breathing phase, arXiv:1903.10251) rather than against ground truth, and say so.
3. **Name them honestly** — "physics-derived proxy for fine crackle", never "fine crackle". Naming a
   proxy after the thing it proxies is how the ICBHI-metric problem started.

### Recommendation

**No / defer — unless Dr. Khan can name a specific clinician this week.** Rationale: the annotation
is orthogonal (roadmap §G0 — it raises the ceiling, it doesn't choose Path A vs B), it is revisitable
at G6, and it would run parallel through Week 4 in direct competition with the two *mandatory* engine
fixes (M35 → concept extractors, M13 → bottleneck head) across a three-person team.

An unvalidated annotation set lowers credibility more than a missing one does.

### The 15-minute agenda

1. Can you name a clinician who will validate ~200 labeled cycles? *(If no → decision is No, stop.)*
2. If yes: what turnaround, and will they arbitrate disagreements between our labelers?
3. Confirm: we proceed label-free, name fine-grained concepts as proxies, and revisit at G6 if
   results are strong enough that the Q1 ceiling becomes the binding constraint.

---

## G1 — Pre-code kill-search ✅ **STILL OPEN — proceed to build**

**Searched 2026-08-15.** Five queries run; five closest candidates fetched and checked.

| Query | Closest hit | Twin? |
|---|---|---|
| physics-derived acoustic concept bottleneck respiratory/lung sound | wavelet/resonance DSP classification; MobileNet "bottleneck blocks" | **No** — architectural bottleneck, not concept bottleneck |
| concept bottleneck model respiratory auscultation 2026 | Context-Aware CBM for ARDS (arXiv:2508.09719, MLHC 2025) | **No** — EHR + clinical notes + Llama-3 on MIMIC-IV. No audio. |
| label-free concept discovery lung sound crackle/wheeze | Label-free CBM (Oikarinen et al., ICLR) | **No** — general method, LLM/CLIP-derived concepts, not audio, not physics |
| interpretable respiratory sound classification | HISET (PMC10404967) | **No** — verified by fetch: ensemble feature engineering. No bottleneck, no leakage measure, no intervention. |
| concept leakage / faithfulness audit respiratory | Leakage-controlled paediatric EHR model (BMC 2026) | **No** — *data* leakage in EHR text, a different sense of the word |

**Verdict: no close twin. The specific intersection — physics/DSP-derived, clinically-named acoustic
concepts in a strict bottleneck over lung-sound audio, with leakage measurement and intervention —
remains unoccupied.** Build the shared engine.

HISET is the strongest evidence *for* the gap: it is the closest interpretable respiratory-audio work
and it explicitly does none of the three things (no bottleneck, no leakage, no intervention).

### ⚠️ Three naming collisions to handle in the write-up

Not twins, but terms already claimed. Using them loosely invites a reviewer to conflate our work with
prior art — the same class of problem as the ICBHI-metric and "official split" mislabels.

1. **"Label-free CBM" is taken** (Oikarinen et al., ICLR) and means *LLM/CLIP-derived* concepts.
   Ours are *physics/DSP-derived*. Never write "label-free concept bottleneck" unqualified — say
   **"physics-derived"** or **"signal-derived"**, and cite theirs as the contrast.
2. **"Leakage" in respiratory ML already means data/label leakage** (BMC 2026). Ours is CBM *concept*
   leakage — information bypassing the bottleneck. Always qualify it as **"concept leakage"** and
   define it on first use.
3. **CBM leakage measurement is established method** — the ARDS paper reports "completeness scores"
   for exactly this. So the *technique* is not novel; the *respiratory-audio instantiation* is. Frame
   it the way `Novelty Search.md` §1 already frames cross-task disagreement: domain-specific
   combination, not invention. Cite Koh et al. 2020 and the ARDS CBM up front.

---

## Annotation effort estimate — for the physician conversation

Asif has potential clinician access via family. This is the estimate to take to them. **The headline
number is ~2 hours of clinician time, not a labeling marathon** — but only if the protocol is
designed right.

### ⚠️ Superseded — see `CLINICIAN_LABELING_PACK.md`

The estimate below assumed DSP pre-labeling to cut clinician effort. **That protocol is invalid for
validation**: if the clinician sees the DSP's answer first they anchor on it, and the "agreement" we
measure is partly agreement with our own suggestion. Circular.

Corrected plan: **120 clips labeled BLIND, ~75 min**. Fewer clips, but they actually validate
something. Full protocol, folder layout, instruction sheet and quality controls are in
`CLINICIAN_LABELING_PACK.md`.

| Protocol | Clips | Clinician time | Valid? |
|---|---|---|---|
| ~~Pre-labeled review~~ | 300 | ~2 h | ❌ circular |
| **Blind labeling** | **120** | **~75 min** | ✅ |

### The realistic ask — 2 sittings, ~2 hours total

1. **Calibration, ~45 min.** Go through 20–30 cycles together and agree the rubric: what counts as
   fine vs. coarse crackle, how to mark inspiratory vs. expiratory phase, wheeze pitch bands. *This
   is the session that determines whether the labels are worth anything* — it produces the written
   labeling guide.
2. **Validation pass, ~60 min.** They review the 200 pre-labeled cycles and correct what's wrong.
3. *(Optional, ~20 min)* Arbitrate disagreements if two students label independently — which you want
   anyway, to report inter-rater agreement.

**Student-side work (parallel, doesn't consume clinician time):** ~1 day to build the labeling tool +
DSP pre-labels, then ~3–4 h for the student pre-labeling pass.

### Ask them these three things

1. **Can you give ~2 hours across two sittings in the next 2–3 weeks?**
2. **Are you comfortable judging crackle type from a recording rather than at the bedside?** — Ask
   this explicitly. Recorded auscultation is genuinely different from bedside, and some clinicians
   will (reasonably) say they can't reliably call fine vs. coarse from a clip. **A "no" here is
   valuable information, not a failure** — it would tell you the fine-grained concepts are not
   reliably human-labelable either, which reframes them as proxies for everyone, not just for us.
3. **May we name you as the validating clinician?** Needed for the "validated against a clinical
   reference" claim to mean anything.

### Scope note worth stating plainly

The 150–300 cycles are a **validation subset, not training data.** Their job is to check whether the
DSP extractors measure what they claim — not to train the model. That is why the number is small and
the ask is bounded, and it is worth saying to the physician so they don't picture an open-ended
commitment.

### Timing — this does not block anything

G1 has cleared, so **engine build (M35 concept extractors) can start now**. Labeling runs parallel
through Week 4 per the build plan. Taking a day or two to ask the family physicians costs nothing on
the critical path. If the answer turns out to be no, the label-free mitigations in the G0 section
above still apply.

---

## Protocol freeze — status

| Item | Status |
|---|---|
| Official split committed | ✅ `Asif's/ICBHI_challenge_train_test.txt` (920 recordings, 539/381) |
| Split loader + leakage policy | ✅ `Asif's/audit/official_split.py` — patients 156, 218 leak; policy documented in Protocol §1 |
| Official ICBHI metric | ✅ `icbhi_score_official` in every results JSON; `Asif's/audit/icbhi_score_audit.py` |
| Confusion matrix required | ✅ Protocol §1 essential #9 |
| CIs + paired test required | ✅ Protocol §1 essential #10; `Asif's/Statistics/` |
| **CC1 — raw per-patient score dumping** | ✅ `Asif's/Statistics/owmtl_scores.py` — canonical schema + validated paired DeLong / McNemar / bootstrap CI (39/39 tests). Generators splice `SCORE_DUMP_CELL`. |

---

## Week 0 verdict — **GO**

**3 of 4 items clear.**

- **G1 clear** — no twin found. This was the blocking gate for engine code, and it has passed.
  **M35 → concept extractors can start now.**
- **G4 answered** — device axis dropped; SPRSound pediatric shift becomes the sole covariate stress.
- **G0 open but non-blocking** — clinician being arranged (1–2 days). Orthogonal to the build per
  roadmap §G0; revisitable at G6.
- **CC1 done** — canonical score format + the paired tests it unlocks, validated 39/39. In
  validation the paired test was **~9x tighter than the unpaired fallback** (SE 0.0086 vs 0.0785),
  turning p=0.57 into p<0.0001 on identical data. That is the power the project has been leaving on
  the table in every comparison so far.

**Nothing is blocking engine code. M35 → concept extractors (Gate G2) is next.**
