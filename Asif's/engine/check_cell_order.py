#!/usr/bin/env python3
"""
Static execution-order check for notebooks.

    python3 check_cell_order.py nb1.ipynb [nb2.ipynb ...]

WHY THIS EXISTS
---------------
`check_undefined_names.py` catches a name read before it is bound. It cannot catch a *resource*
used before it is set up, because the names all resolve fine. That gap shipped a real bug:

    M49 cell  7  globbed /content/drive/MyDrive/** looking for the checkpoint
    M49 cell 14  mounted Drive

Every name was defined, every cell compiled, `check_undefined_names.py` was clean — and the run
died with "checkpoint candidates found: 0" after the user had waited for a 2-hour training run to
finish. The path simply did not exist yet.

This pass walks the cells in order and enforces ordering rules between *producers* and
*consumers* of a resource. Each rule is (name, produces, consumes): a cell matching `consumes`
must not appear before a cell matching `produces`.

Exit code 1 if any rule is violated, so it can gate a commit.
"""
import json
import re
import sys

# (rule name, producer pattern, consumer pattern, consumer exemption)
RULES = [
    ("Drive must be mounted before any cell reads a Drive path",
     r"OWMTL_DRIVE_OUTPUT_V2|drive\.mount\(",
     r"/content/drive/",
     r"OWMTL_DRIVE_OUTPUT_V2"),

    ("DATA_ROOT/SPLIT_FILE must be resolved before they are read",
     r"OWMTL_DATA_SETUP_V2|^DATA_ROOT\s*=",
     r"\bDATA_ROOT\b|\bSPLIT_FILE\b",
     r"OWMTL_DATA_SETUP_V2"),

    ("CFG must exist before it is redirected or read",
     r"^CFG\s*=\s*\{",
     r'\bCFG\[|\bCFG\.get\(',
     None),
]


def code_cells(nb):
    return [(i, "".join(c["source"])) for i, c in enumerate(nb["cells"])
            if c["cell_type"] == "code"]


def check(path):
    nb = json.load(open(path))
    cells = code_cells(nb)
    problems = []

    for name, prod_pat, cons_pat, exempt in RULES:
        prod = next((i for i, s in cells if re.search(prod_pat, s, re.M)), None)
        if prod is None:
            continue                      # rule does not apply to this notebook
        for i, s in cells:
            if i >= prod:
                break
            if exempt and re.search(exempt, s):
                continue
            if re.search(cons_pat, s, re.M):
                line = next((l.strip() for l in s.splitlines()
                             if re.search(cons_pat, l)), "")
                problems.append(f"  cell {i} uses it, but cell {prod} sets it up\n"
                                f"      rule: {name}\n"
                                f"      line: {line[:88]}")
    return problems


def main(paths):
    bad = 0
    for p in paths:
        probs = check(p)
        short = p.split("/")[-1]
        if probs:
            bad += 1
            print(f"### {short}: {len(probs)} ORDERING PROBLEM(S)")
            for x in probs:
                print(x)
        else:
            print(f"### {short}: cell order OK")
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
