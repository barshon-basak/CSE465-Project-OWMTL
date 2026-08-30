"""OWMTL concept-engine package.

Submodules are imported explicitly (`from owmtl.leakage import estimate_leakage`).
They are deliberately NOT eagerly imported here: bottleneck/intervention/m2_* need
torch, and eagerly pulling them in made the torch-free modules (icbhi_data, leakage,
device_check, concept_extractors) untestable anywhere without a GPU stack installed.
"""
