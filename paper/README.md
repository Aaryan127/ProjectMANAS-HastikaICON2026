# Manuscript

`hastika_system_paper.tex` is the draft of the system-description paper, and `paper_refs.bib`
lists every work it cites. The paper describes the submitted systems, the 12-model Task A and
8-model Task B Gemma-4-12B TAPT ensembles, and reports their official test scores from
[`results/final_scores.json`](../results/final_scores.json).

Every number in the draft traces to [docs/experiments.md](../docs/experiments.md), a run record
under [`results/experiments/`](../results/experiments/),
[`results/final_scores.json`](../results/final_scores.json), the class counts of the organizer
files, or arithmetic on those. The paper names no run numbers; [docs/experiments.md](../docs/experiments.md) maps each comparison to its
runs, and [docs/methods.md](../docs/methods.md) describes the system. Appendix A tells the
development story in order for both tasks, and Appendix B lists the ideas that were prepared but
produced no result (MuRIL-large, Qwen3-8B and others).

Before submission:

- The paper is in ACL 2023 format, camera-ready mode. `ACL2023.sty` and `acl_natbib.bst` in this
  folder are the files of the ACL 2023 template (the same git blobs, `0d0d4d8809` and
  `ca569042cb`, as independent copies of that template), so the folder builds as is, locally
  (`pdflatex`, `bibtex`, `pdflatex` twice) or on Overleaf. Authors: Robin Punnen Alex and Aaryan
  Ganesh Salunke, Manipal Institute of Technology, Manipal Academy of Higher Education. If the
  shared-task review is anonymous, switch to `\usepackage[review]{ACL2023}` and replace the
  repository links with an anonymized mirror.
- Every reference was verified against an authoritative record, and the text was checked for
  overlap with its sources; [checks/README.md](checks/README.md) has both, and the tools to
  rerun them.
- No overview paper for the shared task was found; it is cited through the organizers'
  starting kit (`hastika-2026-starting-kit`). Swap in the overview once it is published.
- The abstract links the repository (its `main` branch, this full record); the Section 8 footnote
  links the `reproduce` branch, which contains only what is needed to rebuild the submitted systems.
- In ACL 2023 format, everything except the reference list (main text,
  Limitations, Ethics, Reproducibility and both appendices) fits in 9 pages; with the 63
  references the PDF is 13 pages. Past ICON calls allowed unlimited references; confirm that
  for the 2026 shared task. Recheck the length after any edit.
- Appendix A's remarks on how the team felt at each stage come from the project's records and
  the authors' working sessions (for example, the first-week note that the SVM "surprisingly"
  beat MuRIL). The authors should check them against their own memory.
