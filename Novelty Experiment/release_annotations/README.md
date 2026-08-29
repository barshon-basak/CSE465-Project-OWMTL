# ICBHI clinician annotations, v1

132 respiratory cycles from ICBHI 2017, independently labelled by one clinician who was
blind to the ICBHI annotations. To our knowledge these are the first fine-grained
(fine-vs-coarse crackle) reference labels released for this corpus.

## Files
- `ICBHI_clinician_annotations_v1.csv` - the labels, joined to the ICBHI reference.

## Provenance
Clips were sampled from ICBHI 2017 by `Asif's/engine/build_listening_pack.py` (seed 42),
stratified over normal / crackle-only / wheeze-only / both. **The sample is not random:**
cycles shorter than 0.9 s were excluded and each stratum was sorted longest-first, so these
clips are systematically longer, and easier to judge, than the corpus average. Any
detector scored on this subset will look better here than on ICBHI as a whole.

The rater listened with headphones after calibrating on reference exemplars, could replay
freely, and was instructed to answer `unsure` rather than guess. 132 clips were
presented (120 unique cycles plus hidden repeats);
116 carry an answer. The repeats are used to measure self-consistency.

## Schema
`clinician_crackles` : none | fine | coarse | both | unsure | ambiguous
`clinician_wheeze`   : none | wheeze | rhonchi | both | unsure | ambiguous
`clinician_confidence`   : low | medium | high
`clinician_audio_quality`: ok | noisy | unusable

`ambiguous` is not in the task's declared vocabulary but appears 12 times; it means the
rater could not decide and should be treated as `unsure`. Blank rows are clips the rater
marked `unusable` and declined to score - that is a deliberate answer, not missing data.

## Measured reliability
| quantity | crackles | wheeze |
|---|---|---|
| vs ICBHI, Cohen's kappa | 0.035 [-0.1571, 0.2101] | 0.266 [0.0357, 0.4437] |
| vs ICBHI, raw agreement | 0.5278 | 0.7182 |
| sensitivity vs ICBHI | 0.2308 | 0.2973 |
| specificity vs ICBHI | 0.8036 | 0.9315 |
| intra-rater kappa (hidden duplicates) | 0.7143 | 0.6 |

The rater is far more consistent with themselves than with ICBHI. That gap is the reason
this file exists.

## Limitations - read before using
1. **One rater.** No inter-rater agreement can be computed. Published multi-rater studies
   (n=7 and n=12) carry the reliability argument; this adds granularity, not consensus.
2. **Not a random sample** - see Provenance. Do not compute a corpus-level detector score
   from it.
3. **Fine vs coarse is barely populated** (108 usable crackle answers, of which only a
   handful are coarse). The distinction is released as a reference, not as a benchmark.
4. These labels are an *independent second opinion*, not a correction of ICBHI. Where they
   disagree, neither is established as right.

## Citation
Released as part of the OWMTL corrected-evaluation study. Cite the ICBHI database
(Rocha et al., Physiol. Meas. 40(3), 2019) alongside any use of these labels.
