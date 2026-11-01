"""Scan the paper for markers associated with LLM-generated text.

Markers, with their sources:
  * excess vocabulary: the 291 style words whose frequency rose sharply in 2024 abstracts
    (Kobak et al., "Delving into ChatGPT usage in academic writing through excess vocabulary",
    arXiv:2406.07016, Fig. S6), plus the vocabulary list of Wikipedia's "Signs of AI writing";
  * phrases and structures from that guide: significance inflation ("serves as", "plays a
    crucial role"), negative parallelism ("not X but Y", "not only ... but also", "rather than"),
    didactic disclaimers ("it is important to note"), summary openers ("In summary", "Overall"),
    em dashes, bold inline headers in prose;
  * colon and semicolon density, and the spread of sentence lengths (uniform length is a
    weaker sign; human scientific prose varies).
Single hits are weak evidence; the guide's advice is to look for markers that stack.

    python paper/checks/stylecheck.py paper/hastika_system_paper.tex
    python paper/checks/stylecheck.py --summary $(git ls-files '*.md' '*.ipynb')
"""
import re
import statistics
import sys

KOBAK = """accentuates acknowledges acknowledging addresses adept adhered adhering advancement advancing
advancements advocates advocating affirming afflicted aiding akin align aligning aligns alongside amidst
assessments attains attributed augmenting avenue avenues bolster bolstered bolstering broader burgeoning
capabilities capitalizing categorized categorizes categorizing combating commendable compelling complicates
complicating comprehending comprising consequently consolidates contributing conversely correlating crafted
crafting culminating customizing delineates delve delved delves delving demonstrating dependability
dependable detailing detrimentally diminishes diminishing discern discerned discernible discerning
displaying disrupts distinctions distinctive elevate elevates elevating elucidate elucidates elucidating
embracing emerges emphasises emphasising emphasize emphasizes emphasizing employing employs empowers
emulating emulation enabling encapsulates encompass encompassed encompasses encompassing endeavors
endeavours enduring enhancements enhances ensuring equipping escalating evaluates evolving exacerbating
examines exceeding excels exceptional exceptionally exerting exhibiting exhibits expedite expediting
exploration explores facilitated facilitates facilitating featuring formidable fostering fosters
foundational furnish garnered garnering gauged grappling groundbreaking groundwork harness harnesses
harnessing heighten heightened hinder hinges hinting hold holds illuminates illuminating imbalances
impacting impede impeding imperative impressive inadequately incorporates incorporating influencing
inherent initially innovative inquiries integrates integrating integration interconnectedness interplay
intricacies intricate intricately introduces invaluable investigates involves juxtaposed leverages
leveraging maintaining merges methodologies meticulous meticulously multifaceted necessitate
necessitates necessitating necessity notable noteworthy nuanced nuances offering optimizing orchestrating
outlines overlook overlooking paving persist pinpoint pinpointed pinpointing pioneering pioneers pivotal
poised pose posed poses posing predominantly preserving pressing promise pronounced propelling
realm realms recognizing refine refines refining remarkable renowned revealing reveals
revolutionize revolutionizing revolves scrutinize scrutinized scrutinizing seamless seamlessly seeks
serves serving shaping shedding showcased showcases showcasing signifying solidify spanned spanning
spurred stands stemming strategically streamline streamlined streamlines streamlining struggle
substantiated substantiates surged surmount surpass surpassed surpasses surpassing swift swiftly
thorough transformative typically ultimately uncharted uncovering underexplored underscore underscored
underscores underscoring unexplored unlocking unparalleled unraveling unveil unveiled unveiling unveils
uphold upholding urging utilizes varying versatility warranting yielding""".split()
WIKI = """additionally boasts crucial landscape robust showcase tapestry testament valuable vibrant
garner highlight highlights highlighting enhance enhancing insights leverage foster comprehensive
notably potential findings particularly delving seamless holistic""".split()
PHRASES = [r"serves? as", r"stands? as", r"plays? an? (crucial|key|pivotal|vital|important) role",
           r"it is (important|worth) (to note|noting)", r"in summary", r"in conclusion", r"^overall,",
           r"not only\b", r"\bnot just\b", r"\bnot\b[^.;:]{1,60}, but\b", r"\brather than\b",
           r"a testament", r"in the realm", r"deep dive", r"key (insight|takeaway)", r"sheds? light",
           r"paves? the way", r"underscores?", r"highlights? the importance", r"evolving landscape"]


def detex(text):
    text = text.split("\\begin{document}", 1)[-1]
    text = re.sub(r"(?m)^%.*$", "", text)
    text = re.sub(r"\\(cite[pt]?|citealp|ref|label|url|footnote)\{[^}]*\}", " ", text)
    text = re.sub(r"\\(textit|emph|texttt|section\*?|subsection)\{([^}]*)\}", r" \2 ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", text)
    return re.sub(r"[{}$&\\]", " ", text)


def read(path):
    raw = open(path, errors="ignore").read()
    if path.endswith(".ipynb"):
        import json
        raw = "\n".join("".join(c["source"]) for c in json.loads(raw)["cells"] if c["cell_type"] == "markdown")
    if path.endswith(".tex"):
        return raw.split("\\begin{document}", 1)[-1], detex(raw)
    text = re.sub(r"```.*?```", " ", raw, flags=re.S)          # code blocks are not prose
    text = re.sub(r"`[^`]*`", " ", text)
    return raw, text


def summary(paths):
    vocab = sorted(set(KOBAK) | set(WIKI))
    for path in paths:
        body, text = read(path)
        words = re.findall(r"[a-z]+", text.lower())
        hits = [w for w in words if w in vocab]
        phrases = [p for p in PHRASES if re.search(p, text, re.I | re.M)]
        dashes = body.count("---") if path.endswith(".tex") else body.count("\u2014")
        if hits or phrases or dashes:
            print(f"{path}: {len(words)} words; vocab {sorted(set(hits))}; phrases {phrases}; em dashes {dashes}")


def main(path):
    body, text = read(path)
    words = re.findall(r"[a-z]+", text.lower())
    vocab = sorted(set(KOBAK) | set(WIKI))
    hits = {w: words.count(w) for w in vocab if words.count(w)}
    print(f"{len(words)} words; marker vocabulary: {sum(hits.values())} hits "
          f"({1000 * sum(hits.values()) / len(words):.1f} per 1,000 words)")
    for w, n in sorted(hits.items(), key=lambda kv: -kv[1]):
        print(f"  {w}: {n}")
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text)) if len(s.split()) > 2]
    print("phrases and structures:")
    for pattern in PHRASES:
        found = [s for s in sentences if re.search(pattern, s, re.I | re.M)]
        if found:
            print(f"  /{pattern}/: {len(found)}")
            for s in found:
                print(f"      - {s[:150]}")
    lengths = [len(s.split()) for s in sentences]
    bold = len(re.findall(r"\\textbf\{", body))
    print(f"em dashes: {body.count('---')}; bold inline headers: {bold}")
    print(f"colons per 100 sentences: {100 * sum(s.count(':') for s in sentences) / len(sentences):.0f}; "
          f"semicolons per 100 sentences: {100 * sum(s.count(';') for s in sentences) / len(sentences):.0f}")
    print(f"sentence length: mean {statistics.mean(lengths):.1f}, sd {statistics.stdev(lengths):.1f}, "
          f"coefficient of variation {statistics.stdev(lengths) / statistics.mean(lengths):.2f}")


if __name__ == "__main__":
    if sys.argv[1] == "--summary":
        summary(sys.argv[2:])
    else:
        main(sys.argv[1])
