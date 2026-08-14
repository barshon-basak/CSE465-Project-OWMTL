# Shared split artifacts

Two files belong here, and **both must be committed**:

```
cycles_index_v1.csv    one row per annotated respiratory cycle (~6,898 rows)
split_v1.json          patient -> partition assignment, for both tasks
```

They are not in the repo yet because generating them requires a local copy of
ICBHI 2017. Generate once, commit, and do not regenerate without bumping to `v2`
— every member's numbers are comparable only because all four read these exact
files.

```bash
python scripts/build_index.py --data-root <ICBHI root> --out splits/
python tests/test_icbhi.py          # 17 contract tests, no dependencies
```

`build_index.py` refuses to report success if the split fails validation, so a
clean run plus a green test suite is sufficient evidence to commit.

## What "v1" pins

- `seed=42`, `test_frac=0.40`, `calib_frac=0.15`
- The 19 unseen-disease patients held out of every training partition, every task
- Asthma (n=1) and LRTI (n=2) excluded from the disease task, retained as
  sound-event training data
- The ICBHI challenge split preserved alongside, when the challenge split file is
  present in the dataset copy

Bump to `v2` — never edit `v1` in place — if any of those change. Anyone holding
results against `v1` needs to know their numbers moved.
