"""CPU checks of the reproduction package: notebooks, session plan and the session workflow.

Training is replaced by a fake launcher, so these tests cover orchestration only, not GPU
training, model access or accuracy.
"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import pandas as pd

from hastika import reproduce
from hastika.common.paths import PROJECT_ROOT
from hastika.common.tasks import FILES
from hastika.evaluation.ensemble import write_bundle

spec = importlib.util.spec_from_file_location("build_notebooks", PROJECT_ROOT / "reproduce/build_notebooks.py")
build_notebooks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_notebooks)


class ReproducePackageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_committed_notebooks_match_the_generator(self):
        expected = build_notebooks.build()
        committed = {p.name for p in (PROJECT_ROOT / "reproduce/kaggle").glob("*.ipynb")}
        self.assertEqual(committed, set(expected))
        for name, nb in expected.items():
            on_disk = json.loads((PROJECT_ROOT / "reproduce/kaggle" / name).read_text())
            self.assertEqual(on_disk, json.loads(json.dumps(nb)), name)
            self.assertEqual(on_disk["nbformat"], 4)
            for cell in on_disk["cells"]:
                self.assertIn(cell["cell_type"], ("markdown", "code"))
                if cell["cell_type"] == "code":
                    self.assertEqual(cell["outputs"], [])
                    compile("".join(cell["source"]), name, "exec")
                    self.assertNotIn("hf_", "".join(cell["source"]).lower().replace("hf_token", ""))

    def test_session_plan_matches_the_submitted_recipes(self):
        plan = reproduce.load_sessions()
        for task in ("a", "b"):
            recipe = json.loads((PROJECT_ROOT / f"configs/task_{task}/submitted.json").read_text())
            sessions = plan[task]["sessions"]
            self.assertEqual([s["group"] for s in sessions], [g["id"] for g in recipe["groups"]])
            for s, g in zip(sessions, recipe["groups"]):
                self.assertEqual(s["classifiers"], len(g["classifier_seeds"]))
                self.assertEqual(s["rdrop"], g["rdrop"])
                self.assertEqual(s["tapt_seed"], g["tapt_seed"])
                self.assertLess(s["hours"], 12)
            self.assertEqual(sum(s["classifiers"] for s in sessions), recipe["member_count"])
            official = json.loads((PROJECT_ROOT / "results/final_scores.json").read_text())[task]["macro_f1"]
            self.assertAlmostEqual(plan[task]["official_macro_f1"], official, places=6)
            holdout = json.loads((PROJECT_ROOT / f"configs/task_{task}/holdout.json").read_text())
            self.assertEqual([g["id"] for g in holdout["groups"]], [g["id"] for g in recipe["groups"]])
            self.assertTrue(set(plan[task]["recorded_holdout"]) <= {g["id"] for g in recipe["groups"]})

    def test_find_file_checks_the_recorded_checksum(self):
        good, bad = self.root / "good", self.root / "bad"
        (good / "nested").mkdir(parents=True)
        bad.mkdir()
        (bad / "x.csv").write_text("wrong\n")
        (good / "nested" / "x.csv").write_text("right\n")
        digest = reproduce.sha256(good / "nested" / "x.csv")
        self.assertEqual(reproduce.find_file("x.csv", digest, [bad, good]), good / "nested" / "x.csv")
        with self.assertRaises(FileNotFoundError):
            reproduce.find_file("x.csv", "0" * 64, [bad, good])

    def test_sessions_collect_and_finish_without_training(self):
        task = "a"
        raw = self.root / "raw"
        raw.mkdir()
        pd.DataFrame({"id": ["001", "002"], "Comment": ["test one", "test two"]}).to_csv(raw / FILES[task]["test"], index=False)
        external = self.root / "external.csv"
        pd.DataFrame({"Comment": ["external example"]}).to_csv(external, index=False)
        manifest = json.loads((PROJECT_ROOT / "data/sources.json").read_text())

        def fake_prepare(task, raw, data):
            data.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"id": ["t1", "t2"], "Comment": ["one", "two"], "Label": ["Hate", "Non-Hate"]}).to_csv(data / "all.csv", index=False)
            return {"fixture": True}

        def fake_launch(cmd, log, gpu=None):
            destination = Path(cmd[cmd.index("--out") + 1])
            if "hastika.models.llm_tapt" in cmd:
                destination.mkdir(parents=True)
                return 0
            prediction = Path(cmd[cmd.index("--predict") + 1])
            write_bundle(destination / prediction.stem, [[.8, .2], [.3, .7]], ["001", "002"], task)
            return 0

        def fake_stage(task, dest, roots=None):
            return raw, external

        sessions_work = self.root / "sessions"
        with patch("hastika.reproduce.stage", side_effect=fake_stage), \
             patch("hastika.runner.prepare", side_effect=fake_prepare), \
             patch("hastika.runner.sha256", return_value=manifest["external"]["sha256"]), \
             patch("hastika.runner.launch", side_effect=fake_launch):
            for group in reproduce.group_ids(task):
                reproduce.run_session(task, group, sessions_work, execute=True, gpus=[0, 1])
            out = reproduce.output_root(sessions_work, task, "submitted")
            self.assertFalse((out / "data").exists())          # comment data scrubbed
            self.assertFalse((out / "external_text.csv").exists())
            # A finished group is skipped; an unfinished one is refused.
            reproduce.run_session(task, "bag1", sessions_work, execute=True, gpus=[0, 1])
            (out / "bag2" / "group_metadata.json").unlink()
            with self.assertRaises(FileExistsError):
                reproduce.run_session(task, "bag2", sessions_work, execute=True)
            reproduce.run_session(task, "bag2", self.root / "redo", execute=True)
            # Gather the sessions' outputs into a fresh folder, as the combine notebook does.
            combined = self.root / "combined"
            missing = reproduce.collect(task, combined, roots=[sessions_work, self.root / "redo"])
            self.assertEqual(missing, [])
            report = reproduce.finish(task, combined)
        self.assertEqual(report["members"], 12)
        with zipfile.ZipFile(report["predictions_zip"]) as archive:
            frame = pd.read_csv(archive.open("predictions.csv"), dtype=str)
        self.assertEqual(frame.id.tolist(), ["001", "002"])
        self.assertEqual(frame.label.tolist(), ["Non-Hate", "Hate"])

    def test_smoke_runs_every_stage_without_training(self):
        task = "b"
        raw = self.root / "raw"
        raw.mkdir()
        pd.DataFrame({"id": ["9", "8", "7"], "Comment": ["one", "two", "three"]}).to_csv(raw / FILES[task]["test"], index=False)
        external = self.root / "external.csv"
        pd.DataFrame({"Comment": [f"external {i}" for i in range(100)]}).to_csv(external, index=False)
        manifest = json.loads((PROJECT_ROOT / "data/sources.json").read_text())
        classes = ["Gender", "Geo-political", "Others", "Political", "Religion", "Violence"]

        def fake_prepare(task, raw, data):
            data.mkdir(parents=True, exist_ok=True)
            rows = [{"id": f"t{i}", "Comment": f"text {i}", "Hate Category": classes[i % 6]} for i in range(300)]
            pd.DataFrame(rows).to_csv(data / "all.csv", index=False)
            return {}

        seen = []

        def fake_launch(cmd, log, gpu=None):
            seen.append(cmd)
            destination = Path(cmd[cmd.index("--out") + 1])
            if "hastika.models.llm_tapt" in cmd:
                destination.mkdir(parents=True)
                (destination / "adapter_config.json").write_text("{}")
                return 0
            if "hastika.inference.predict" in cmd:
                write_bundle(destination, [[.5, .1, .1, .1, .1, .1]] * 3, ["9", "8", "7"], task)
                return 0
            Path(log).write_text("initialised from TAPT adapter\n")
            write_bundle(destination / Path(cmd[cmd.index("--predict") + 1]).stem,
                         [[.5, .1, .1, .1, .1, .1]] * 3, ["9", "8", "7"], task)
            return 0

        with patch("hastika.reproduce.check_environment", return_value=[]), \
             patch("hastika.reproduce.stage", side_effect=lambda task, dest, roots=None: (raw, external)), \
             patch("hastika.runner.prepare", side_effect=fake_prepare), \
             patch("hastika.runner.sha256", return_value=manifest["external"]["sha256"]), \
             patch("hastika.runner.launch", side_effect=fake_launch):
            checks = reproduce.smoke(task, self.root / "work", gpus=[0, 1])
        self.assertTrue(all(checks.values()))
        classifier = [c for c in seen if "hastika.models.llm_classifier" in c]
        self.assertEqual(len(classifier), 2)
        self.assertTrue(all("--no-collapse-check" in c and c[c.index("--epochs") + 1] == "1" for c in classifier))
        self.assertTrue(any("hastika.inference.predict" in c for c in seen))
        audit = json.loads((self.root / "work/task_b_smoke/data_audit.json").read_text())
        self.assertEqual(audit["smoke"], {"train_rows": 96, "external_rows": 64})
        self.assertFalse((self.root / "work/data").exists())

    def test_smoke_recipes_match_the_submitted_model_settings(self):
        for task in ("a", "b"):
            smoke = json.loads((PROJECT_ROOT / f"configs/task_{task}/smoke.json").read_text())
            full = json.loads((PROJECT_ROOT / f"configs/task_{task}/submitted.json").read_text())
            for key in ("model", "lr", "head_lr", "lora_rank", "prompt", "class_weight", "label_smoothing",
                        "bs", "grad_accum", "max_len", "tapt_bs", "tapt_grad_accum", "tapt_max_len"):
                self.assertEqual(smoke[key], full[key], key)
            self.assertEqual(smoke["mode"], "full")


if __name__ == "__main__":
    unittest.main()
