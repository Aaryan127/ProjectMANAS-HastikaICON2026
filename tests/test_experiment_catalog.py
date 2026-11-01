"""The experiment record is complete, consistently numbered and runnable from this repository."""
import hashlib
import json
import re
import unittest

from hastika.common.paths import PROJECT_ROOT

REPO_URL = "https://github.com/Aaryan127/ProjectMANAS-HastikaICON2026.git"
RUNS_PER_TASK = 38


class ExperimentCatalogue(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads((PROJECT_ROOT / "results/experiment_catalog.json").read_text())
        cls.entries = cls.catalog["experiments"]

    def test_runs_are_numbered_without_gaps(self):
        for task in ("a", "b"):
            runs = sorted({e["run"] for e in self.entries if e["task"] == task})
            self.assertEqual(runs, list(range(1, RUNS_PER_TASK + 1)), task)
        keys = [(e["task"], e["run"], e["session"]) for e in self.entries]
        self.assertEqual(len(keys), len(set(keys)))

    def test_every_notebook_is_catalogued_and_matches_its_record(self):
        on_disk = {str(p.relative_to(PROJECT_ROOT))
                   for p in (PROJECT_ROOT / "notebooks/experiments").glob("task_*/*.ipynb")}
        catalogued = {e["notebook"] for e in self.entries if e["notebook"]}
        self.assertEqual(on_disk, catalogued)
        for entry in self.entries:
            with self.subTest(task=entry["task"], run=entry["run"], session=entry["session"]):
                record = json.loads((PROJECT_ROOT / entry["record"]).read_text())
                for key in ("task", "run", "session", "status", "outcome", "notebook"):
                    self.assertEqual(record[key], entry[key])
                self.assertTrue(entry["notebook"] or record["script"], "a run needs a notebook or a script")

    def test_notebook_metadata_code_hash_and_cleared_outputs(self):
        for entry in self.entries:
            if not entry["notebook"] or (entry["task"], entry["run"]) == ("a", 10):
                continue
            with self.subTest(notebook=entry["notebook"]):
                nb = json.loads((PROJECT_ROOT / entry["notebook"]).read_text())
                meta = nb["metadata"]["hastika"]
                self.assertEqual((meta["task"], meta["run"], meta["session"], meta["record"]),
                                 (entry["task"], entry["run"], entry["session"], entry["record"]))
                code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
                body = "\n".join("".join(c["source"]) for c in code_cells)
                self.assertEqual(hashlib.sha256(body.encode()).hexdigest(), meta["code_sha256"])
                for cell in code_cells:
                    self.assertEqual(cell["outputs"], [])
                    self.assertIsNone(cell["execution_count"])
                header = "".join(nb["cells"][0]["source"])
                self.assertRegex(header.splitlines()[0], rf"\bRuns? {entry['run']}\b")

    def test_notebooks_clone_this_repository_and_stage_data(self):
        for path in (PROJECT_ROOT / "notebooks/experiments").glob("task_*/*.ipynb"):
            with self.subTest(notebook=path.name):
                nb = json.loads(path.read_text())
                code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
                urls = set(re.findall(r"https://github\.com/[\w.-]+/[\w.-]+?\.git", code))
                self.assertEqual(urls, {REPO_URL})
                self.assertIn("hastika.research.data", code)
                self.assertNotIn("raw.githubusercontent.com", code)
                for module in re.findall(r"hastika\.(?:common|models|task_a|task_b)\.\w+", code):
                    self.fail(f"{module}: research modules live under hastika.research")

    def test_final_ensemble_members_match_the_configs(self):
        for task, total in (("a", 12), ("b", 8)):
            members = {(e["run"], e["session"]): e["final_ensemble_members"]
                       for e in self.entries if e["task"] == task and e["final_ensemble_members"]}
            config = json.loads((PROJECT_ROOT / f"configs/task_{task}/submitted.json").read_text())
            from_config = {(g["run"], g.get("session")): len(g["classifier_seeds"]) for g in config["groups"]}
            self.assertEqual(members, from_config)
            self.assertEqual(sum(members.values()), total)

    def test_metric_extracts_are_bounded(self):
        for entry in self.entries:
            record = json.loads((PROJECT_ROOT / entry["record"]).read_text())
            for log in record["logs"]:
                self.assertEqual(len(log["sha256"]), 64)
                self.assertTrue(log["log"].startswith(f"run{entry['run']:02d}_") or
                                log["log"].startswith(f"run{entry['run']}_"), log["log"])
                for metric in log["metrics"]:
                    self.assertGreater(metric["line"], 0)
                    self.assertTrue(0 <= metric["macro_f1"] <= 1)
                    if metric["accuracy"] is not None:
                        self.assertTrue(0 <= metric["accuracy"] <= 1)
                    for score in metric["per_class_f1"].values():
                        self.assertTrue(0 <= score <= 1)
                for comparison in log["comparisons"]:
                    self.assertTrue(0 <= comparison["p_better"] <= 1)
                    self.assertLessEqual(comparison["ci95"][0], comparison["ci95"][1])


if __name__ == "__main__":
    unittest.main()
