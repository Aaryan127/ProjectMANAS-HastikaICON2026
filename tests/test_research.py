"""CPU checks for the research package's data handling (no GPU libraries needed)."""
import pathlib
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from hastika.common.tasks import FILES
from hastika.research import data
from hastika.research.task_b import corpora


class ResearchData(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)

    def test_phase_inputs_are_the_validation_file_without_labels(self):
        rows = {"id": ["7", "3", "9"], "Comment": ["first comment", "second", "third one"],
                "Label": ["Hate", "Non-Hate", "Hate"]}
        pd.DataFrame(rows).to_csv(self.root / FILES["a"]["validation"], index=False)
        out = data.write_phase_inputs("a", self.root)
        written = pd.read_csv(out, dtype=str)
        self.assertEqual(out.name, "binary_validation_inputs.csv")
        self.assertEqual(written.columns.tolist(), ["id", "Comment"])
        self.assertEqual(written.id.tolist(), rows["id"])
        self.assertEqual(written.Comment.tolist(), rows["Comment"])

    def test_unlabelled_text_comes_only_from_task_b_training_text(self):
        raw = self.root / "raw"
        raw.mkdir()
        for name in ["multiclass_train.csv", "binary_train.csv", "multiclass_validation_inputs.csv",
                     "hastika_multiclass_test.csv"]:
            pd.DataFrame({"id": ["1"], "Comment": [f"text from {name}"]}).to_csv(raw / name, index=False)
        external = self.root / "external.csv"
        pd.DataFrame({"Comment": ["external kannada comment"]}).to_csv(external, index=False)
        with patch.object(corpora, "RAW_DATA_DIR", raw):
            texts = corpora.load([str(raw / "multiclass_train.csv"), str(external)],
                                 min_words=1, verbose=False)
            self.assertEqual(len(texts), 2)
            for name in ["binary_train.csv", "multiclass_validation_inputs.csv", "hastika_multiclass_test.csv"]:
                with self.subTest(name=name), self.assertRaises(SystemExit):
                    corpora.load([str(raw / name)], verbose=False)


if __name__ == "__main__":
    unittest.main()
