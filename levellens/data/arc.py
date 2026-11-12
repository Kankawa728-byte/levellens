"""ARC adapter: school science questions with a school grade, 7 levels (grades 3 to 9).

The grade is only in the original AI2 release. That archive is 681 MB, almost all of it a
text corpus we do not use, so the adapter reads only the six question files from it (2.3 MB).
The filter follows the data preparation of the published comparison
(github.com/arthur-thuy/qde-ordinality, MIT, `src/tools/data_manager/_arc_data_manager.py`):
multiple-choice questions without a diagram and with exactly four options.

Splits: the released Train, Dev and Test files, Easy and Challenge together.
Extra split `train_balanced`: the two most common grades (5 and 8) cut to 500 questions each,
as in the published papers. It is used only for replication runs and for the majority-row test.
"""

from __future__ import annotations

from pathlib import Path

from levellens.data.schema import DatasetInfo, make_item
from levellens.data.splits import capped_per_level

GRADES = (3, 4, 5, 6, 7, 8, 9)
INFO = DatasetInfo(
    name="arc",
    level_names=tuple(f"grade {grade}" for grade in GRADES),
    trained=True,
    source="AI2 ARC-V1-Feb2018.zip",
)

URL = "https://ai2-public-datasets.s3.amazonaws.com/arc/ARC-V1-Feb2018.zip"
FOLDER_IN_ARCHIVE = "ARC-V1-Feb2018-2"
FILE_SPLITS = {"train": "Train", "dev": "Dev", "test": "Test"}
SETS = ("Easy", "Challenge")
NEEDED_COLUMNS = ("questionID", "AnswerKey", "isMultipleChoiceQuestion", "includesDiagram", "schoolGrade", "question")
OPTION_MARKS = (("(A) ", "(B) ", "(C) ", "(D) ", "(E) "), ("(1) ", "(2) ", "(3) ", "(4) ", "(5) "))
BALANCED_CAPS = {GRADES.index(5): 500, GRADES.index(8): 500}
BALANCED_SEED = 0


def csv_name(subset: str, split: str) -> str:
    """Return the file name of one question file, for example ARC-Easy-Train.csv."""
    return f"ARC-{subset}-{FILE_SPLITS[split]}.csv"


def download(raw_dir: Path) -> Path:
    """Fetch the six question files from the AI2 archive unless they are already there. Return their folder.

    `remotezip` reads single files out of a remote zip with HTTP range requests,
    so the 1.4 GB text corpus in the same archive is never downloaded.
    """
    folder = raw_dir / "arc"
    wanted = [(subset, split) for subset in SETS for split in FILE_SPLITS]
    if all((folder / csv_name(subset, split)).exists() for subset, split in wanted):
        return folder

    from remotezip import RemoteZip

    folder.mkdir(parents=True, exist_ok=True)
    print(f"arc: reading the question files from {URL}")
    with RemoteZip(URL) as archive:
        for subset, split in wanted:
            member = f"{FOLDER_IN_ARCHIVE}/ARC-{subset}/{csv_name(subset, split)}"
            content = archive.read(member)
            if len(content) != archive.getinfo(member).file_size:
                raise OSError(f"arc: {member} arrived incomplete")
            part = folder / (csv_name(subset, split) + ".part")
            part.write_bytes(content)
            part.replace(folder / csv_name(subset, split))
    return folder


def split_options(text: str) -> tuple[str, list[str]] | None:
    """Split "stem (A) one (B) two (C) three (D) four" into the stem and four options.

    Options are marked with letters or with digits. Return None unless there are exactly four.
    """
    for marks in OPTION_MARKS:
        if all(mark in text for mark in marks[:4]) and marks[4] not in text:
            stem, rest = text.split(marks[0], 1)
            options = []
            for mark in marks[1:4]:
                option, rest = rest.split(mark, 1)
                options.append(option.strip())
            options.append(rest.strip())
            return stem.strip(), options
    return None


def _answer_index(key: str) -> int:
    """Turn the answer key ("A".."D" or "1".."4") into an index from 0."""
    key = str(key).strip()
    return int(key) - 1 if key.isdigit() else ord(key) - ord("A")


def _read(folder: Path, split: str) -> list[dict]:
    """Read the Easy and Challenge files of one split, apply the filter and return items."""
    import pandas as pd

    items = []
    for subset in SETS:
        path = folder / csv_name(subset, split)
        frame = pd.read_csv(path)
        missing = [column for column in NEEDED_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"arc: {path.name} lacks the columns {missing}")
        empty = frame[list(NEEDED_COLUMNS)].isna().sum()
        if empty.any():
            raise ValueError(f"arc: {path.name} has empty cells: {empty[empty > 0].to_dict()}")
        frame = frame[(frame["isMultipleChoiceQuestion"] == 1) & (frame["includesDiagram"] == 0)]
        for row in frame.to_dict("records"):
            parsed = split_options(row["question"])
            if parsed is None:
                continue
            stem, options = parsed
            grade = int(row["schoolGrade"])
            if grade not in GRADES:
                raise ValueError(f"arc: question {row['questionID']} has grade {grade}, outside {GRADES}")
            items.append(
                make_item(
                    id=row["questionID"],
                    info=INFO,
                    split=split,
                    question=stem,
                    options=options,
                    level=GRADES.index(grade),
                    meta={
                        "grade": grade,
                        "set": subset,
                        "exam": str(row.get("examName", "")),
                        "year": str(row.get("year", "")),
                        "answer": _answer_index(row["AnswerKey"]),
                    },
                )
            )
    return items


def build(raw_dir: Path) -> dict[str, list[dict]]:
    """Fetch ARC if needed and return the train, dev, test and train_balanced items."""
    folder = download(raw_dir)
    splits = {split: _read(folder, split) for split in FILE_SPLITS}
    splits["train_balanced"] = capped_per_level(splits["train"], BALANCED_CAPS, seed=BALANCED_SEED)
    return splits
