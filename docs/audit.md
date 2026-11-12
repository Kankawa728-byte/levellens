# Final-model error audit

中文版：[audit.zh-CN.md](audit.zh-CN.md)

**Owner:** Wenyue. **Status on 12 November 2026:** protocol and worksheet prepared; final-model cases not yet supplied by F. No case has been adjudicated. Counts of model error, label noise and unclear are **not available**, rather than zero. This document must not be cited as a completed human audit.

## Handoff required from F

Provide five test errors for each of RACE++, ARC and MATH, selected from a model frozen using development-set performance. Each row needs dataset, item ID, run ID, split, true level, predicted level, probability vector or regression output, and absolute ordinal error. Include the complete question, options and available context in a private file, not in this public repository. Preserve RACE++ passages and MATH solutions as review context, and identify what the model actually saw.

The list should be ordered by descending absolute difference between predicted and true level. Resolve ties by item ID so that selection is reproducible. F should retain the source prediction-file hash and frozen run ID. B checks IDs and labels against `load_split(dataset, "test")`. Selecting cases from test is post-hoc error analysis, not permission to tune on test.

## Decision rules

| Category | Evidence needed |
|---|---|
| model error | The source label is plausible for its source exam or level, the model misses it, and no concrete data defect explains the case. Record the reasoning demand or cue that may have been missed. |
| label noise | A concrete source-label inconsistency, source metadata error or corrupted item supports a label concern. “It feels easier” is insufficient. Preserve the original label and describe the evidence; do not relabel test data. |
| unclear | The source level cannot be adjudicated reliably, necessary context is unavailable to the model/reviewer, or plausible explanations conflict. Record the missing evidence. |

These categories are operational judgements, not causal proof. Use a secondary flag for omitted passage, omitted hint, duplicated options, markup, or conflicting identical inputs. Missing input context alone is not proof of label noise. Since labels are target levels, a reviewer cannot infer a student's success probability from the text alone.

Wenyue reads each full case and writes a specific one-sentence reason. A second member checks disputed cases if available. Record the actual reviewer and date; do not name a reviewer who has not reviewed the case.

## Fifteen-case worksheet

“Pending” is a workflow state, not one of the three judgement categories. Replace each row only when its evidence arrives.

| Slot | Dataset | Item ID | Run ID | True / predicted level | Category | Reason / evidence | Reviewer / date |
|---|---|---|---|---|---|---|---|
| RACE-1 | RACE++ | Pending | Pending | Pending | Pending | Awaiting F | Pending |
| RACE-2 | RACE++ | Pending | Pending | Pending | Pending | Awaiting F | Pending |
| RACE-3 | RACE++ | Pending | Pending | Pending | Pending | Awaiting F | Pending |
| RACE-4 | RACE++ | Pending | Pending | Pending | Pending | Awaiting F | Pending |
| RACE-5 | RACE++ | Pending | Pending | Pending | Pending | Awaiting F | Pending |
| ARC-1 | ARC | Pending | Pending | Pending | Pending | Awaiting F | Pending |
| ARC-2 | ARC | Pending | Pending | Pending | Pending | Awaiting F | Pending |
| ARC-3 | ARC | Pending | Pending | Pending | Pending | Awaiting F | Pending |
| ARC-4 | ARC | Pending | Pending | Pending | Pending | Awaiting F | Pending |
| ARC-5 | ARC | Pending | Pending | Pending | Pending | Awaiting F | Pending |
| MATH-1 | MATH | Pending | Pending | Pending | Pending | Awaiting F | Pending |
| MATH-2 | MATH | Pending | Pending | Pending | Pending | Awaiting F | Pending |
| MATH-3 | MATH | Pending | Pending | Pending | Pending | Awaiting F | Pending |
| MATH-4 | MATH | Pending | Pending | Pending | Pending | Awaiting F | Pending |
| MATH-5 | MATH | Pending | Pending | Pending | Pending | Awaiting F | Pending |

| Category | Count |
|---|---|
| model error | Not available |
| label noise | Not available |
| unclear | Not available |
| reviewed cases | 0 of 15 |

## Three slide examples

Select one informative case per dataset after review. Each example must show source level, model prediction, decision and one short reason, with the item ID and run ID in speaker notes. Slots: RACE++ pending; ARC pending; MATH pending. The duplicate-input examples in [datasets.md](datasets.md) are data-quality observations, not substitutes for the final model's worst errors.

## Completion check

- All 15 IDs belong to the frozen test splits and are wrong predictions.
- All 15 rows have an actual reviewer, date and reason.
- The three category counts total 15; the pending count is zero.
- The three examples are drawn from the reviewed table.
- No audit count is used as a population error-rate estimate; the cases were selected for severity.
- Slides and narration are updated only after the table is complete.
