# Reference and originality checks

Run on 2026-10-31 against the version of `paper/hastika_system_paper.tex` in this commit. Rerun
both checks after substantive edits.

## References

`paper_refs.bib` has 63 entries and the paper cites all 63 (no uncited or missing keys).
Each entry was checked against an authoritative record of the work, not copied from memory:

| entry | verified against |
|---|---|
| `berg-kirkpatrick-etal-2012-empirical` | ACL Anthology BibTeX record |
| `bohra-etal-2018-dataset` | ACL Anthology BibTeX record |
| `caselli-etal-2021-hatebert` | ACL Anthology BibTeX record |
| `chakravarthi-etal-2021-findings-shared` | ACL Anthology BibTeX record |
| `chakravarthi-etal-2022-dravidiancodemix` | Crossref record, doi:10.1007/s10579-022-09583-7 |
| `chen-etal-2016-training` | arXiv API record (title, authors, date) |
| `conneau-etal-2020-unsupervised` | ACL Anthology BibTeX record |
| `cortes-vapnik-1995-support` | Crossref record, doi:10.1007/BF00994018 |
| `das-etal-2022-data` | Crossref record, doi:10.1145/3511095.3531277 |
| `davidson-etal-2017-automated` | Crossref record, doi:10.1609/icwsm.v11i1.14955 |
| `dettmers-etal-2022-llmint8` | arXiv API record of the paper (title, authors, date); venue as published |
| `dettmers-etal-2023-qlora` | arXiv API record (title, authors, date) |
| `devlin-etal-2019-bert` | ACL Anthology BibTeX record |
| `dietterich-2000-ensemble` | Crossref record, doi:10.1007/3-540-45014-9_1 |
| `dodge-etal-2020-fine` | arXiv API record (title, authors, date) |
| `dror-etal-2018-hitchhikers` | ACL Anthology BibTeX record |
| `efron-1979-bootstrap` | Crossref record, doi:10.1214/aos/1176344552 |
| `fan-etal-2008-liblinear` | JMLR listing (volume, pages) |
| `fortuna-nunes-2018-survey` | Crossref record, doi:10.1145/3232676 |
| `gala-etal-2023-indictrans2` | arXiv API record of the paper (title, authors, date); venue as published |
| `gemma-team-2026-gemma4` | arXiv API record (title, authors, date) |
| `gururangan-etal-2020-dont` | ACL Anthology BibTeX record |
| `hande-etal-2020-kancmd` | ACL Anthology BibTeX record |
| `hastika-2026-starting-kit` | organizers' repository README |
| `he-etal-2023-debertav3` | arXiv API record of the paper (title, authors, date); venue as published |
| `howard-ruder-2018-universal` | ACL Anthology BibTeX record |
| `hu-etal-2022-lora` | arXiv API record (title, authors, date) |
| `joachims-1998-text` | Crossref record, doi:10.1007/BFb0026683 |
| `kavatagi-rachh-2025` | Crossref record, doi:10.1007/s10579-025-09836-1 |
| `khanuja-etal-2021-muril` | arXiv API record (title, authors, date) |
| `kim-wurster-emoji` | package repository (authors), version from uv.lock |
| `kingma-ba-2015-adam` | arXiv API record (title, authors, date) |
| `koehn-2004-statistical` | ACL Anthology BibTeX record |
| `kudo-richardson-2018-sentencepiece` | ACL Anthology BibTeX record |
| `lakshminarayanan-etal-2017-simple` | arXiv API record of the paper (title, authors, date); venue as published |
| `liang-etal-2021-rdrop` | arXiv API record of the paper (title, authors, date); venue as published |
| `lin-etal-2017-focal` | Crossref record, doi:10.1109/ICCV.2017.324 |
| `loshchilov-hutter-2017-sgdr` | arXiv API record (title, authors, date) |
| `loshchilov-hutter-2019-decoupled` | arXiv API record (title, authors, date) |
| `madhani-etal-2023-aksharantar` | ACL Anthology BibTeX record |
| `mangrulkar-etal-2022-peft` | the PEFT repository's own citation block |
| `mehdad-tetreault-2016-characters` | ACL Anthology BibTeX record |
| `micikevicius-etal-2018-mixed` | arXiv API record (title, authors, date) |
| `miyato-etal-2017-adversarial` | arXiv API record (title, authors, date) |
| `mosbach-etal-2021-stability` | arXiv API record (title, authors, date) |
| `nayak-joshi-2022-l3cube` | ACL Anthology BibTeX record |
| `pascanu-etal-2013-difficulty` | arXiv API record of the paper (title, authors, date); venue as published |
| `paszke-etal-2019-pytorch` | arXiv API record of the paper (title, authors, date); venue as published |
| `pedregosa-etal-2011-scikit` | JMLR listing (volume, pages) |
| `platt-2000-probabilistic` | MIT Press volume listings (chapter pp. 61-74, volume 2000) |
| `polyak-juditsky-1992-acceleration` | Crossref record, doi:10.1137/0330046 |
| `sap-etal-2019-risk` | ACL Anthology BibTeX record |
| `sparck-jones-1972-statistical` | Crossref record, doi:10.1108/eb026526 |
| `speer-2019-ftfy` | Crossref record, doi:10.5281/zenodo.2591652 |
| `sridhar-1990-kannada` | library catalogue records and author profile |
| `srivastava-etal-2014-dropout` | JMLR listing (volume, pages) |
| `sun-etal-2019-fine` | Crossref record, doi:10.1007/978-3-030-32381-3_16 |
| `szegedy-etal-2016-rethinking` | Crossref record, doi:10.1109/CVPR.2016.308 |
| `waseem-hovy-2016-hateful` | ACL Anthology BibTeX record |
| `wolf-etal-2020-transformers` | ACL Anthology BibTeX record |
| `wu-etal-2016-google` | arXiv API record (title, authors, date) |
| `yang-etal-2025-qwen3` | arXiv API record (title, authors, date) |
| `zhang-etal-2021-revisiting` | arXiv API record (title, authors, date) |

Notes from the check:

- **Gemma 4.** The report is arXiv:2607.02770. Its tokenizer section ("a SentencePiece
  tokenizer", "byte-level encodings", 262k entries) supports the paper's statement that emoji never
  become unknown tokens for Gemma; it also lists the 26B (mixture-of-experts) and 31B models.
- **Sridhar (1990)** is cited only for Kannada being agglutinative. The paper's description of
  *-beku* as necessitative is not credited to it, because its treatment of that suffix was not
  checked.
- **Platt.** The chapter (pp. 61-74) appeared in the 2000 MIT Press volume; it is often cited as
  1999.
- **Software** (PyTorch, Transformers, PEFT, bitsandbytes, scikit-learn, ftfy, emoji) is cited
  through the publication or citation block each project asks for.
- **Shared task.** No overview paper for HASTIKA @ ICON-2026 was found; the task is cited through
  the organizers' starting-kit repository. Replace that entry with the overview paper once it is
  published.

## Originality

`textcheck.py` compares the paper and the documentation with the texts they draw on.
`fetch_sources.sh` rebuilds the public part of the corpus; the HASTIKA dataset paper must be added
from your own copy. Source texts are not committed.

Corpus used (32 texts): the HASTIKA dataset paper; the full texts of R-Drop, LoRA, QLoRA, Zhang
et al. (re-initialization), focal loss, MuRIL, FGM (Miyato et al.), Dodge et al., Mosbach et al.,
AdamW, the Gemma 4 report, Gururangan et al. (TAPT), hateBERT, the DravidianLangTech-2021 shared-task report,
KanCMD, Mehdad and Tetreault, Koehn, Berg-Kirkpatrick et al., Waseem and Hovy, Bohra et al. and
HingBERT; the abstracts of every cited arXiv paper; the organizers' two READMEs; five other teams'
public HASTIKA READMEs; and the model cards of MuRIL, Sarvam-1, Sarvam-Translate, hateBERT and
HingRoBERTa.

| check | target | result |
|---|---|---|
| shared runs of 6+ words | paper | none |
| shared runs of 5+ words | paper | 2, both generic: "it has not yet been"; "NF4 with double quantization and" (the cited method's name) |
| shared runs of 6+ words | 104 Markdown files and notebooks | 19, all URLs, the six category names (the task's label set and our prompt), or one stock sentence ("please cite the HASTIKA dataset paper and the"), which was then reworded |
| closest source sentence (Jaccard of content words) | paper, 240 sentences | highest 0.53, the task definition listing the six categories; every other sentence 0.36 or lower |
| closest source sentence | README, reproduce/README, methods, reproducibility, paper/README | highest 0.34 |
| exact-phrase web search | 16 of the paper's most generic sentences (definitions of TAPT, R-Drop, QLoRA, WordPiece, the bootstrap, calibration, mojibake, ensembles, the ethics statement, the reproducibility section) | no match for any |

The checker was validated by planting copied passages (from the R-Drop paper and the organizers'
README) in a test file; it reported both, at their full length.

Limits: the corpus covers the works the paper paraphrases most and every cited abstract, not the
full text of every reference, and a web search covers only indexed pages. Neither test judges
whether an idea is credited; the paper cites a source wherever it describes another work's method,
data or result.

```bash
bash paper/checks/fetch_sources.sh /tmp/sources          # then add hastika_dataset_paper.txt
python paper/checks/textcheck.py exact --sources /tmp/sources paper/hastika_system_paper.tex
python paper/checks/textcheck.py exact --sources /tmp/sources --n 5 paper/hastika_system_paper.tex
python paper/checks/textcheck.py paraphrase --sources /tmp/sources paper/hastika_system_paper.tex
```

## Writing style

`stylecheck.py` scans the paper and every Markdown file and notebook for markers associated with
LLM-generated text: the 291 excess style words of Kobak et al. (arXiv:2406.07016, Fig. S6), the
vocabulary and phrase list of Wikipedia's "Signs of AI writing" (significance inflation, negative
parallelism such as "not X but Y" and its "Y, in place of X" variants, didactic disclaimers,
summary openers),
em dashes and bold inline headers.

| target | result |
|---|---|
| paper | no marker phrases, no em dashes, no bold inline headers; the only listed word is "stemming", used for the NLP technique |
| all Markdown files and notebooks | the same: only "stemming", in the experiment log |

```bash
python paper/checks/stylecheck.py paper/hastika_system_paper.tex
python paper/checks/stylecheck.py --summary $(git ls-files '*.md' '*.ipynb')
```
