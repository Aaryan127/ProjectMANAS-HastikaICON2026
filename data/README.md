# Data

The [organizer starting kit](https://github.com/shankarb14/Hastika-ICON2026) and the
[competition](https://www.codabench.org/competitions/17784/) are the sources. The organizers'
[data-use note](https://github.com/shankarb14/Hastika-ICON2026/blob/main/LICENSE_NOTE.md) allows
research and non-commercial use and forbids redistribution, so no comments or comment-derived
files are in this repository.

## Organizer files

Put these in `data/raw/`:

| file | rows | role |
|---|---:|---|
| `binary_train.csv` | 6,446 | Task A training labels |
| `hastika_binary_validation.csv` | 806 | Task A released validation labels |
| `hastika_binary_test.csv` | 806 | Task A test inputs |
| `multiclass_train.csv` | 3,159 | Task B training labels |
| `hastika_multiclass_validation.csv` | 395 | Task B released validation labels |
| `hastika_multiclass_test.csv` | 396 | Task B test inputs |

[`sources.json`](sources.json) records the SHA-256 of each file, and the tools refuse a file that
differs. A re-encoded or newer organizer release needs a new source manifest and a comparison of
the split ids; changing a hash alone does not make two releases equivalent.

Only these six files are needed to reproduce the submitted systems. Do not use the 395-row
`multiclass_validation_inputs.csv` of the validation phase in place of the 396-row Task B test
file; they are different phases.

## Holdouts

```bash
hastika-prepare --task a     # writes data/derived/task_a_combined_holdout/{all,train,holdout}.csv
hastika-prepare --task b
```

Each command combines that task's training and released validation files, drops duplicate ids,
collapses agreeing duplicate texts, drops groups with conflicting labels, and splits off a
stratified 15% holdout with seed 42.

| task | labelled rows | holdout rows | holdout fingerprint |
|---|---:|---:|---|
| A | 7,193 | 1,079 | `815110ff24` |
| B | 3,532 | 530 | `f85f4f049b` |

The holdouts share no normalized text with their training partitions. The organizer-defined
labelled sets share no ids with the test inputs. A few comments recur as text, 6 distinct texts
for A and 4 for B, and `split.json` reports these counts.

The reproduction package does this itself before training (see
[`reproduce/README.md`](../reproduce/README.md)).

## External Kannada text

TAPT reads the processed `offenseval_kn.csv`, built from `community-datasets/offenseval_dravidian`
(Kannada) by the authors' research code (on the `main` branch). Put an authorized copy in
`data/external/`; its checksum and its 3,247-row count are in `sources.json`. Processing kept
targeted insults and `Not_offensive` rows and dropped untargeted and non-Kannada rows,
short or mostly native-script comments, self-duplicates, and comments matching the HASTIKA files
available at the time. The upstream dataset revision was not recorded, so a rebuild is not
guaranteed to match byte for byte. Obtain the processed file from the authors, or rebuild it and
record a new manifest. Public availability of the upstream dataset does not grant redistribution
of HASTIKA text.

Only the `Comment` column is used. Before TAPT, the final-system runner removes external comments
whose normalized text matches a holdout or test input. One external comment matches a Task A test
comment.

`data/raw/`, `data/external/`, `data/derived/` and runtime audits are ignored by Git.
