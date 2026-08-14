"""
OWMTL shared pipeline — Group 5, CSE465.

Maintained by Asif Mahbub (Member A: shared representation + AST backbone).
Every member imports this package so that all four workstreams train on
byte-identical partitions and report byte-identical metric definitions.

Quick start
-----------
    from owmtl import icbhi, features

    split = icbhi.load_split("splits/split_v1.json")
    index = icbhi.read_cycle_index("splits/cycles_index_v1.csv")

    train = icbhi.select_cycles(index, split, partition="train")
    test  = icbhi.select_cycles(index, split, partition="test")

    # Member B/C/D disease-task partitions come out of the same file:
    known_train  = icbhi.disease_patients(split, "known_train")
    unknown_eval = icbhi.disease_patients(split, "unknown_eval")

Regenerate the split artifacts with:
    python scripts/build_index.py --data-root <ICBHI root> --out splits/

Only `icbhi` is dependency-free. `features` needs librosa/scipy, `separability`
needs scikit-learn/scipy, `reporting` needs scikit-learn/matplotlib, and the AST
front-end additionally needs torchaudio.
"""

__version__ = "1.0.0"

from . import icbhi  # noqa: F401  (dependency-free, always safe to import)

__all__ = ["icbhi", "features", "separability", "reporting"]
