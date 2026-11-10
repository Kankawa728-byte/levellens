# Share B · Literature, data checks and audit

中文版：[B-literature-data-audit.zh-CN.md](B-literature-data-audit.zh-CN.md)

**Owner:** Wenyue · **Buddy:** D · **Size:** 24 h Tier 1 + 4 h Tier 2

The evidence around the models: what the data really look like, what has been published, and whether our worst errors are the model's fault or the labels' fault. This share also keeps the slides consistent and prepares the documents we hand in.

## Files you own

- `docs/datasets.md` (dataset cards), `docs/related_work.md`, `docs/audit.md`
- `notebooks/data_checks.ipynb`
- The slide template, the storyboard and the reflection document. These live on the shared drive, not in this repository.

## Tier 1

| Id | Task | Done when | Hours | Due |
|---|---|---|---|---|
| B1 | Data checks and dataset cards for RACE++, ARC, MATH and ScienceQA (text only). For each dataset: items per split and per level against [docs/design.md](../docs/design.md); question length per level; exact and near-duplicate questions inside a split and across splits; 20 items read per level; source and licence. | `docs/datasets.md` has one card per dataset with one class-balance figure. Every mismatch with the published counts is reported in the issue of share A. | 5.5 | Counts: Wed 11 Nov 23:59. Cards: Thu 12 Nov 20:00 |
| B2 | Published baselines and related work. Protocol notes for Benedetto (2023) and Thuy et al. (2025): splits, the balanced ARC subset, input format, metrics, number of seeds. The baseline table checked number by number against the arXiv versions. References with confirmed venues. | `docs/related_work.md` has the protocol notes and the checked table. The reference slide is in the deck. | 4 | Thu 12 Nov 23:59 |
| B3 | Human audit of the 30 worst errors of the final models, 10 per dataset. For each one: read the question, decide "model error", "label noise" or "unclear", and write one line why. | `docs/audit.md` has the table, the count per category and three example cases for the slides. | 3.5 | Sat 14 Nov 09:00 |
| B4 | Slide template, storyboard, recording guide and recording schedule | The template is shared on Thu 12 Nov at 12:00. Storyboard v1 lists every slide with its speaker, content and seconds on Fri 13 Nov at 12:00. The recording guide in [docs/video.md](../docs/video.md) is checked and completed. | 3.5 | Fri 13 Nov 12:00 |
| B5 | Proofreading of all slides and figure captions; the limitations slide | Every slide says "target level" and never claims difficulty measured on students. Every number on a slide matches the tables. | 2 | Sat 14 Nov 12:00 |
| B6 | Reflection document: collect each member's lines, build the contribution table, collect the signatures | One document, signed by all six. It is kept outside this public repository. | 1.5 | Sat 14 Nov 22:00 |
| B7 | Submission checklist and peer-evaluation reminders | The checklist is run with the lead before the upload. Everyone is reminded when the peer evaluation opens. | 1 | Sun 15 Nov |

For B1 the lead provides the loading code (task A2). You do not write the adapters.

## Tier 2

| Id | Task | Done when | Hours |
|---|---|---|---|
| B8 | GSM8K dataset card | The card is in `docs/datasets.md`. | 0.5 |
| B9 | Audit extended to 60 errors, 20 per dataset | The table in `docs/audit.md` has 60 rows and the counts are updated. | 2 |
| B10 | Background notes and one slide: how question difficulty is set in practice (pretesting on students, item response theory) and why our labels are target levels | The notes are in `docs/related_work.md` with their references, and the slide is in the deck. | 1.5 |

## Others wait for you

- Everyone: the slide template on Thu 12 Nov at 12:00, so that slides can be drafted early.
- A: mismatches in the data counts, as soon as you see them.
- A: the storyboard order for the video assembly.

## You wait for

- A: adapters for the data checks. RACE++ and MATH on Wed 11 Nov at 12:00, ARC and ScienceQA at 18:00.
- F: the list of the 30 worst errors on Fri 13 Nov at 15:00.
- Everyone: final slides on Sat 14 Nov at 10:00 and reflection lines at 18:00.

## Video part

1:00–2:20: the datasets and where their level labels come from, what the data checks found, the published baselines we compare with, and what the audit of our worst errors found.
