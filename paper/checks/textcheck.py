"""Check the paper and the documentation for text taken from their sources.

Two tests, run against a folder of plain-text sources (see fetch_sources.sh):

  exact       every run of N or more consecutive words (default 6, case and punctuation
              ignored) shared with any source
  paraphrase  for every sentence, the most similar source sentence by content-word overlap
              (Jaccard), to catch reworded copying that an exact test misses

    python paper/checks/textcheck.py exact --sources SRC paper/hastika_system_paper.tex README.md
    python paper/checks/textcheck.py paraphrase --sources SRC paper/hastika_system_paper.tex
"""
import argparse
import glob
import json
import os
import re

STOP = set("the a an and or of to in on for with by from as at is are was were be been it its this "
           "that these those we our their his her they them which who whom than then so not no but if "
           "into over under each per one two can could may might also only more most less very such".split())


def words(text):
    return re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", text.lower())


def detex(text):
    text = re.sub(r"(?m)^%.*$", "", text)
    text = re.sub(r"\\(cite[pt]?|citealp|ref|label|url|footnote)\{[^}]*\}", " ", text)
    text = re.sub(r"\\(textit|textbf|emph|texttt|paragraph|section\*?|subsection)\{([^}]*)\}", r" \2 ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", text)
    return re.sub(r"[{}$&\\]", " ", text)


def read_target(path):
    text = open(path, errors="ignore").read()
    if path.endswith(".tex"):
        return detex(text)
    if path.endswith(".ipynb"):
        cells = json.loads(text)["cells"]
        return "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "markdown")
    return text


def sources(folder):
    return {os.path.basename(f): open(f, errors="ignore").read() for f in sorted(glob.glob(folder + "/*.txt"))}


def exact(src, targets, n):
    grams = {name: {tuple(w[i:i + n]) for w in [words(t)] for i in range(len(w) - n + 1)} for name, t in src.items()}
    hits = set()
    for target in targets:
        for sentence in re.split(r"(?<=[.!?:])\s+|\n\s*\n", read_target(target)):
            w = words(sentence)
            for name, g in grams.items():
                flags = [tuple(w[i:i + n]) in g for i in range(len(w) - n + 1)]
                i = 0
                while i < len(flags):
                    if flags[i]:
                        j = i
                        while j + 1 < len(flags) and flags[j + 1]:
                            j += 1
                        hits.add((j - i + n, target, name, " ".join(w[i:j + n])))
                        i = j + 1
                    else:
                        i += 1
    for length, target, name, span in sorted(hits, key=lambda h: (-h[0], h[1])):
        print(f"{length:3d} | {target} | {name} | {span}")
    print(f"{len(hits)} shared spans of {n}+ words")


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text)) if len(s.split()) >= 8]


def content(sentence):
    return {w for w in re.findall(r"[a-z]+", sentence.lower()) if w not in STOP and len(w) > 2}


def paraphrase(src, targets, top):
    pool = [(name, s, content(s)) for name, t in src.items() for s in sentences(t)]
    pool = [p for p in pool if len(p[2]) >= 5]
    for target in targets:
        rows = []
        for s in sentences(read_target(target)):
            c = content(s)
            if len(c) >= 5:
                rows.append(max(((len(c & q) / len(c | q), s, name, t) for name, t, q in pool), default=(0, s, "", "")))
        rows.sort(reverse=True)
        print(f"== {target}: {len(rows)} sentences; most similar source sentences")
        for score, s, name, t in rows[:top]:
            print(f"{score:.2f}  OURS: {s[:200]}\n      {name}: {t[:200]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("test", choices=["exact", "paraphrase"])
    parser.add_argument("targets", nargs="+")
    parser.add_argument("--sources", required=True)
    parser.add_argument("--n", type=int, default=6, help="exact: minimum shared run of words")
    parser.add_argument("--top", type=int, default=10, help="paraphrase: pairs shown per file")
    args = parser.parse_args()
    src = sources(args.sources)
    if args.test == "exact":
        exact(src, args.targets, args.n)
    else:
        paraphrase(src, args.targets, args.top)


if __name__ == "__main__":
    main()
