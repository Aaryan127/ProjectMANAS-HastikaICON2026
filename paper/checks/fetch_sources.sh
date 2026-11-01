#!/usr/bin/env bash
# Download the public source texts that paper/checks/textcheck.py compares against.
# Needs curl, pdftotext (poppler) and the GitHub CLI (gh). Usage: bash fetch_sources.sh OUT_DIR
# Texts are for local checking only; do not commit them (they are other authors' work).
# The HASTIKA dataset paper (doi:10.1007/s10579-025-09836-1) is not open access here: add your
# own copy as OUT_DIR/hastika_dataset_paper.txt (pdftotext paper.pdf hastika_dataset_paper.txt).
set -euo pipefail
OUT=${1:?output folder}; mkdir -p "$OUT"; TMP=$(mktemp -d)
arxiv="2106.14448 2106.09685 2305.14314 2006.05987 1708.02002 2103.10730 1605.07725 2002.06305 2006.04884 1711.05101 2607.02770"
acl="2020.acl-main.740 2021.woah-1.3 2021.dravidianlangtech-1.17 2020.peoples-1.6 W16-3638 W04-3250 D12-1091 N16-2013 W18-1105 2022.wildre-1.2"
for id in $arxiv; do curl -sSL -m 120 -o "$TMP/$id.pdf" "https://arxiv.org/pdf/$id" && pdftotext "$TMP/$id.pdf" "$OUT/arxiv_$id.txt"; sleep 3; done
for id in $acl; do curl -sSL -m 60 -o "$TMP/$id.pdf" "https://aclanthology.org/$id.pdf" && pdftotext "$TMP/$id.pdf" "$OUT/acl_$id.txt"; done
# abstracts of every arXiv paper the bibliography cites
ids=$(grep -o 'arXiv:[0-9.]*\|arxiv.org/abs/[0-9.]*' "$(dirname "$0")/../paper_refs.bib" | grep -o '[0-9]\{4\}\.[0-9]*' | sort -u | paste -sd, -)
curl -sSL -m 120 "https://export.arxiv.org/api/query?id_list=$ids&max_results=100" | python3 -c '
import sys, xml.etree.ElementTree as ET
ns = {"a": "http://www.w3.org/2005/Atom"}
for e in ET.parse(sys.stdin).getroot().findall("a:entry", ns):
    print(" ".join(e.find("a:summary", ns).text.split()), end="\n\n")' > "$OUT/arxiv_abstracts.txt"
# the organizers' repositories and other teams' public HASTIKA write-ups
for r in shankarb14/Hastika-ICON2026 shankarb14/SLM-Impact Junaid706/hastika-kanglish-hate-speech \
         SuhaanFrancis/hastika-icon2026-system shahtaz/hastika RifahTasnimJuii/HASTIKA-ICON2026-Kanglish-Hate-Speech; do
  gh api "repos/$r/readme" -H "Accept: application/vnd.github.raw" > "$OUT/readme_${r//\//_}.txt" || true
done
for m in google/muril-base-cased sarvamai/sarvam-1 sarvamai/sarvam-translate GroNLP/hateBERT l3cube-pune/hing-roberta; do
  curl -sSL -m 30 "https://huggingface.co/$m/raw/main/README.md" > "$OUT/card_${m//\//_}.txt" || true
done
rm -rf "$TMP"; ls "$OUT" | wc -l
