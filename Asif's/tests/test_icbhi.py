"""
Self-tests for the split contract. Standard library only — runs on a laptop with
no ICBHI copy and no scientific stack installed.

    python tests/test_icbhi.py

The fixture synthesises an ICBHI-shaped directory with the real patient counts
(64 COPD / 26 Healthy / 14 URTI / 7 Bronchiectasis / 6 Pneumonia /
6 Bronchiolitis / 2 LRTI / 1 Asthma) and empty .wav files, since nothing in
icbhi.py decodes audio.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from owmtl import icbhi  # noqa: E402

DEVICES = ("AKGC417L", "LittC2SE", "Litt3200", "Meditron")
LOCATIONS = ("Tc", "Al", "Ar", "Pl", "Pr", "Ll", "Lr")
CYCLES_PER_RECORDING = 12
RECORDINGS_PER_PATIENT = 5


def make_fixture(root: str) -> None:
    """Write a synthetic ICBHI corpus with the real diagnosis distribution."""
    audio = os.path.join(root, "audio_and_txt_files")
    os.makedirs(audio, exist_ok=True)

    diagnoses = []
    for dx, n in icbhi.EXPECTED_DIAGNOSIS_COUNTS.items():
        diagnoses.extend([dx] * n)
    assert len(diagnoses) == 126

    with open(os.path.join(root, "patient_diagnosis.csv"), "w", encoding="utf-8") as fh:
        for i, dx in enumerate(diagnoses):
            pid = 101 + i
            fh.write(f"{pid},{dx}\n")

            for rec in range(RECORDINGS_PER_PATIENT):
                loc = LOCATIONS[(i + rec) % len(LOCATIONS)]
                dev = DEVICES[(i + rec) % len(DEVICES)]
                stem = f"{pid}_{rec + 1}b1_{loc}_sc_{dev}"
                open(os.path.join(audio, stem + ".wav"), "wb").close()
                with open(os.path.join(audio, stem + ".txt"), "w", encoding="utf-8") as ann:
                    for c in range(CYCLES_PER_RECORDING):
                        start = c * 2.5
                        end = start + 2.4
                        # Rotate through all four (crackle, wheeze) combinations
                        # so every class has support in every partition.
                        crackle = (c + rec) % 2
                        wheeze = ((c + rec) // 2) % 2
                        ann.write(f"{start:.3f}\t{end:.3f}\t{crackle}\t{wheeze}\n")


class SplitContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = tempfile.mkdtemp(prefix="icbhi_fixture_")
        make_fixture(cls.root)
        cls.paths = icbhi.locate(cls.root)
        cls.rows = icbhi.build_cycle_index(cls.paths)
        cls.split = icbhi.build_split(cls.rows, seed=42)
        cls.payload = icbhi.split_to_dict(cls.split, cls.rows, cls.paths)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.root, ignore_errors=True)

    # --- parsing ---------------------------------------------------------

    def test_parse_stem(self):
        m = icbhi.parse_stem("101_1b1_Al_sc_Meditron")
        self.assertEqual(m.patient_id, 101)
        self.assertEqual(m.chest_location, "Al")
        self.assertEqual(m.acquisition_mode, "sc")
        self.assertEqual(m.device, "Meditron")

    def test_parse_stem_rejects_garbage(self):
        with self.assertRaises(ValueError):
            icbhi.parse_stem("not_an_icbhi_name")

    def test_sound_event_label_encoding(self):
        self.assertEqual(icbhi.sound_event_label(0, 0), 0)  # Normal
        self.assertEqual(icbhi.sound_event_label(1, 0), 1)  # Crackle
        self.assertEqual(icbhi.sound_event_label(0, 1), 2)  # Wheeze
        self.assertEqual(icbhi.sound_event_label(1, 1), 3)  # Both

    def test_index_shape(self):
        expected = 126 * RECORDINGS_PER_PATIENT * CYCLES_PER_RECORDING
        self.assertEqual(len(self.rows), expected)
        self.assertEqual(len({r.patient_id for r in self.rows}), 126)

    # --- the invariants everything downstream depends on -----------------

    def test_validate_passes(self):
        problems = icbhi.validate(self.payload, [r.__dict__ for r in self.rows])
        self.assertEqual(problems, [], "\n".join(problems))

    def test_no_patient_in_two_partitions(self):
        seen = set()
        for pids in self.payload["partitions"]["owmtl"]["sound_event"].values():
            overlap = seen & set(pids)
            self.assertEqual(overlap, set(), f"patients in two partitions: {overlap}")
            seen |= set(pids)
        self.assertEqual(len(seen), 126)

    def test_unknown_disease_patients_never_train(self):
        """The anti-leak rule this split exists to enforce."""
        owmtl = self.payload["partitions"]["owmtl"]
        unknown = set(owmtl["disease_role"]["unknown_eval"])
        self.assertEqual(len(unknown), 19)
        trained = set(owmtl["sound_event"]["train"]) | set(owmtl["sound_event"]["calib"])
        self.assertEqual(unknown & trained, set())
        self.assertEqual(unknown, set(owmtl["sound_event"]["held_out"]))

    def test_disease_role_counts(self):
        owmtl = self.payload["partitions"]["owmtl"]["disease_role"]
        known = (
            len(owmtl["known_train"])
            + len(owmtl["known_calib"])
            + len(owmtl["known_test"])
        )
        self.assertEqual(known, 104)          # COPD 64 + Healthy 26 + URTI 14
        self.assertEqual(len(owmtl["unknown_eval"]), 19)
        self.assertEqual(len(owmtl["excluded"]), 3)  # Asthma 1 + LRTI 2

    def test_test_fraction_is_approximately_40_percent(self):
        owmtl = self.payload["partitions"]["owmtl"]["disease_role"]
        frac = len(owmtl["known_test"]) / 104
        self.assertGreater(frac, 0.33)
        self.assertLess(frac, 0.47)

    def test_every_known_disease_appears_in_every_partition(self):
        patients = self.payload["patients"]
        for role in ("known_train", "known_calib", "known_test"):
            pids = self.payload["partitions"]["owmtl"]["disease_role"][role]
            present = {patients[str(p)]["diagnosis"] for p in pids}
            for dx in icbhi.KNOWN_DISEASES:
                self.assertIn(dx, present, f"{dx} missing from {role}")

    def test_split_is_deterministic(self):
        again = icbhi.build_split(self.rows, seed=42)
        self.assertEqual(again.sound_event, self.split.sound_event)
        self.assertEqual(again.disease_role, self.split.disease_role)

    def test_different_seed_gives_different_split(self):
        other = icbhi.build_split(self.rows, seed=7)
        self.assertNotEqual(other.sound_event, self.split.sound_event)

    def test_strict_counts_rejects_truncated_corpus(self):
        subset = [r for r in self.rows if r.patient_id < 200]
        with self.assertRaises(ValueError):
            icbhi.build_split(subset, seed=42)
        icbhi.build_split(subset, seed=42, strict_counts=False)  # override works

    # --- round-tripping through the committed artifacts ------------------

    def test_csv_roundtrip_and_select_cycles(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "cycles_index_v1.csv")
            json_path = os.path.join(tmp, "split_v1.json")
            icbhi.write_cycle_index(self.rows, csv_path)
            icbhi.write_split(self.payload, json_path)

            index = icbhi.read_cycle_index(csv_path)
            split = icbhi.load_split(json_path)
            self.assertEqual(len(index), len(self.rows))
            self.assertIsInstance(index[0]["patient_id"], int)
            self.assertIsInstance(index[0]["start"], float)

            parts = {
                p: icbhi.select_cycles(index, split, partition=p)
                for p in icbhi.SOUND_EVENT_SPLITS
            }
            self.assertEqual(sum(len(v) for v in parts.values()), len(index))
            train_pids = {r["patient_id"] for r in parts["train"]}
            test_pids = {r["patient_id"] for r in parts["test"]}
            self.assertEqual(train_pids & test_pids, set())

            self.assertEqual(len(icbhi.disease_patients(split, "unknown_eval")), 19)

    def test_official_scheme_errors_when_unavailable(self):
        with self.assertRaises(RuntimeError):
            icbhi.select_cycles(
                [r.__dict__ for r in self.rows],
                self.payload,
                partition="test",
                scheme="official",
            )

    def test_official_split_conflicts_are_detected(self):
        """A file-level split that puts one patient in both halves must be caught."""
        stems = sorted({r.stem for r in self.rows})
        bad = {s: ("train" if i % 2 else "test") for i, s in enumerate(stems)}
        split = icbhi.build_split(self.rows, seed=42, official_by_stem=bad)
        payload = icbhi.split_to_dict(split, self.rows, self.paths)
        self.assertTrue(payload["partitions"]["official"]["patients_spanning_both_halves"])
        problems = icbhi.validate(payload, [r.__dict__ for r in self.rows])
        self.assertTrue(any("both halves" in p for p in problems))

    def test_validate_catches_an_injected_leak(self):
        """The leak check must actually fire, not just pass vacuously."""
        import copy

        broken = copy.deepcopy(self.payload)
        owmtl = broken["partitions"]["owmtl"]
        leaked = owmtl["disease_role"]["unknown_eval"][0]
        owmtl["sound_event"]["train"].append(leaked)
        owmtl["sound_event"]["held_out"].remove(leaked)
        problems = icbhi.validate(broken, [r.__dict__ for r in self.rows])
        self.assertTrue(any("leaked" in p for p in problems), problems)


if __name__ == "__main__":
    unittest.main(verbosity=2)
