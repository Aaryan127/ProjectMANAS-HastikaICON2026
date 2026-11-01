"""One place that turns a list of data sources into a list of comments.

Every script that consumes unlabelled text -- tapt.py, extend_vocab.py -- goes
through here, so adding a new corpus to a run is a path on the command line and
nothing else. Accepts:

    data/raw/multiclass_train.csv          a CSV; the text column is auto-detected
    data/external/*.csv                a glob
    scraped/comments.txt               one comment per line
    scraped/dump.jsonl                 one JSON object per line, text auto-detected
    @src/hastika/research/experiments/task_b/corpora.txt    a manifest: one source per line, # comments

POLICY
------
Unlabelled text comes from Task B's own training file, files derived from a
training split, and external corpora. Every other organizer file under
data/raw/ is refused, so validation and test inputs can never reach TAPT or
vocabulary counting. No labels are ever read here.
"""
import glob
import pathlib
import sys

import pandas as pd

from hastika.research.common.paths import PROJECT_ROOT, RAW_DATA_DIR

ROOT = PROJECT_ROOT
RAW_ALLOWED = {"multiclass_train.csv"}
TEXT_COLS = ["Comment", "comment", "text", "Text", "tweet", "content", "sentence"]


def _text_column(df, path):
    for c in TEXT_COLS:
        if c in df.columns:
            return c
    obj = [c for c in df.columns if df[c].dtype == object]
    if not obj:
        sys.exit(f"{path}: no text-like column in {list(df.columns)}")
    return max(obj, key=lambda c: df[c].astype(str).str.len().mean())


def _expand(spec):
    """Manifest, glob, or plain path -> concrete paths."""
    if spec.startswith("@"):
        man = pathlib.Path(spec[1:])
        if not man.is_absolute():
            man = ROOT / man
        out = []
        for line in man.read_text().splitlines():
            line = line.split("#")[0].strip()
            if line:
                out += _expand(line)
        return out
    p = spec if pathlib.Path(spec).is_absolute() else str(ROOT / spec)
    hits = sorted(glob.glob(p))
    if not hits:
        sys.exit(f"no such source: {spec}")
    return [pathlib.Path(h) for h in hits]


def _read(path):
    suf = path.suffix.lower()
    if suf == ".csv":
        df = pd.read_csv(path)
        return df[_text_column(df, path)].astype(str).tolist()
    if suf in (".jsonl", ".ndjson"):
        df = pd.read_json(path, lines=True)
        return df[_text_column(df, path)].astype(str).tolist()
    return [l for l in path.read_text(encoding="utf-8", errors="replace").splitlines()]


def load(specs, clean_fn=None, min_words=2,
         dedupe=True, verbose=True):
    """Resolve sources to a deduplicated list of comments. Reads no labels."""
    texts = []
    for spec in specs:
        for path in _expand(spec):
            if path.resolve().parent == RAW_DATA_DIR.resolve() and path.name not in RAW_ALLOWED:
                sys.exit(f"refusing {path.name}: from data/raw/ only "
                         f"{sorted(RAW_ALLOWED)} may be used as unlabelled text")
            rows = _read(path)
            if clean_fn:
                rows = [clean_fn(t) for t in rows]
            rows = [t for t in rows if len(t.split()) >= min_words]
            if verbose:
                print(f"  {path.relative_to(ROOT) if ROOT in path.parents else path}"
                      f": {len(rows)} rows", flush=True)
            texts += rows
    n_raw = len(texts)
    if dedupe:
        seen, out = set(), []
        for t in texts:                    # a repeated comment is not extra evidence
            k = t.casefold()
            if k not in seen:
                seen.add(k)
                out.append(t)
        texts = out
    if verbose:
        print(f"corpus: {len(texts)} unique comments (from {n_raw})", flush=True)
    return texts
