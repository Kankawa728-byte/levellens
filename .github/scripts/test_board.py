"""Tests of the status board. They need no network and change no file of the repository.

    python .github/scripts/test_board.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("GITHUB_REPOSITORY", "example/levellens")

import board  # noqa: E402

ROOT = board.ROOT
FRIDAY_EVENING = board.when(13, "21:00")


def comment(body: str, created: str, association: str = "NONE") -> dict:
    return {"body": body, "association": association, "created": created}


class Reports(unittest.TestCase):
    def test_done_and_undo_and_the_latest_report_wins(self):
        comments = {"C": [comment("done C1 C2", "2026-11-13T10:00:00Z"), comment("undo C2", "2026-11-13T11:00:00Z"),
                          comment("Done: slides, recording", "2026-11-13T12:00:00Z")]}
        self.assertEqual(board.read_reports(comments), {"C1", "C:slides", "C:recording"})

    def test_a_sentence_that_starts_with_done_changes_nothing(self):
        comments = {"C": [comment("done with the first half of C4", "2026-11-13T10:00:00Z"), comment("I take this", "2026-11-13T10:01:00Z"),
                          comment("> done C1\nquoted mail", "2026-11-13T10:02:00Z")]}
        self.assertEqual(board.read_reports(comments), set())

    def test_only_the_owner_of_the_repository_reports_another_share(self):
        stranger = {"C": [comment("done D1 F2", "2026-11-13T10:00:00Z")]}
        lead = {"C": [comment("done C2 F2\ndone slides", "2026-11-13T10:00:00Z", "OWNER")]}
        self.assertEqual(board.read_reports(stranger), set())
        self.assertEqual(board.read_reports(lead), {"C2", "F2", "C:slides"})

    def test_a_word_of_another_share_is_not_a_task_here(self):
        self.assertEqual(board.read_reports({"B": [comment("done tables", "2026-11-13T10:00:00Z")]}), set())
        self.assertEqual(board.read_reports({"A": [comment("done tables", "2026-11-13T10:00:00Z")]}), {"A:tables"})


class Checks(unittest.TestCase):
    def test_grid_lines_give_the_run_ids_of_the_run_command(self):
        text = "# note\n\ndata=race model=deberta-v3 head=ce seed=0 owner=E model.params.epochs=3\ndata=arc model=distilbert head=corn seed=2\n"
        self.assertEqual(board.grid_run_ids(text), ["race-deberta-v3-ce-s0", "arc-distilbert-corn-s2"])
        self.assertEqual(board.grid_run_ids("data=arc model=distilbert head=ce variant=smoke limit_train=400"), ["arc-distilbert-ce-smoke-n400-s0"])

    def test_the_grid_files_of_the_repository_are_read_in_full(self):
        repository = board.Repository(ROOT)
        self.assertEqual(len(board.grid_run_ids(repository.read(board.TRANSFORMER_GRID))), 20)
        self.assertEqual(len(board.grid_run_ids(repository.read(board.BILSTM_GRID))), 6)

    def test_every_task_and_hand_off_is_known(self):
        owners, pages, titles = board.overview(ROOT, "en")
        self.assertEqual(sorted(owners), list(board.SHARES))
        self.assertTrue(set(board.NUMBERED) <= set(titles))
        tasks = set(board.NUMBERED) | {f"{share}:{word}" for word, (share, *_) in board.EXTRA.items()}
        for name, (source, receivers, task, due, _, _) in board.HANDOFFS.items():
            self.assertTrue(task in board.COMMON if source == "*" else task in tasks, name)
            self.assertEqual(due, board.COMMON[task][0] if source == "*" else due, name)
        for key, (_, _, needs) in board.NUMBERED.items():
            self.assertTrue(set(needs) <= set(board.HANDOFFS), key)

    def test_a_smoke_run_is_not_a_result(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Path(folder) / "results" / "runs.jsonl"
            store.parent.mkdir()
            rows = [{"run_id": "arc-distilbert-ce-smoke-n400-s0", "dataset": "arc", "model": "distilbert", "variant": "smoke-n400"},
                    {"run_id": "arc-r2de-none-s0", "dataset": "arc", "model": "r2de", "variant": ""}]
            store.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            repository = board.Repository(Path(folder))
            self.assertFalse(repository.check(("rows", ["distilbert"], ["arc"]))[0])
            self.assertTrue(repository.check(("rows", ["r2de"], ["arc"]))[0])


def copy_of_the_pages(folder: Path) -> None:
    """Copy what the board reads into an empty folder, without the results."""
    for path in [*board.PAGES.values(), *board.OVERVIEW.values(), board.TRANSFORMER_GRID, board.BILSTM_GRID]:
        (folder / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / path, folder / path)


def facts(text: str) -> dict:
    """Count what must be the same in the English and the Chinese board."""
    return {
        "rows": len(re.findall(r"^\|", text, flags=re.M)),
        "list items": len(re.findall(r"^- ", text, flags=re.M)),
        "task ids": Counter(re.findall(r"\b[A-F]\d{1,2}\b", text)),
        "times": Counter(re.findall(r"\b\d{1,2}:\d\d\b", text)),
        "code": Counter(re.findall(r"`([^`]+)`", text)),
        "clocks": text.count("⏰"),
    }


class Rendering(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        copy_of_the_pages(self.folder)

    def tearDown(self):
        shutil.rmtree(self.folder)

    def make(self, reported=(), rows=(), now=FRIDAY_EVENING, pulls=None) -> board.Board:
        store = self.folder / "results" / "runs.jsonl"
        store.parent.mkdir(exist_ok=True)
        store.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        return board.Board(board.Repository(self.folder), set(reported), pulls, now, root=self.folder)

    def test_an_empty_repository_waits_for_the_grid(self):
        state = self.make()
        name, people = state.bottleneck()
        self.assertEqual(name, "results")
        self.assertEqual(people, set("ABDEF"))  # A, E and F directly; B and D through the tables and the error list
        self.assertTrue(state.blocked(state.tasks["E8"]))
        self.assertFalse(state.blocked(state.tasks["C4"]) and not state.arrived["configs"])
        self.assertIn(("results"), [line[2] for line in state.waits_for("F")])
        self.assertIn(["A", "E", "F"], [line[1] for line in state.waited_by("C")])

    def test_grid_rows_in_the_store_release_the_people_who_wait(self):
        ids = board.grid_run_ids((ROOT / board.TRANSFORMER_GRID).read_text(encoding="utf-8"))
        rows = [{"run_id": run_id, "dataset": run_id.split("-")[0], "model": "x", "variant": ""} for run_id in ids]
        state = self.make(rows=rows)
        self.assertTrue(state.tasks["C5"].done)
        self.assertEqual(state.tasks["C5"].count, (20, 20))
        self.assertFalse(state.blocked(state.tasks["E8"]))
        self.assertNotIn("results", [line[2] for line in state.waits_for("A")])

    def test_outputs_of_models_that_are_not_final_do_not_close_a_task(self):
        audit = self.folder / "results" / "audit" / "worst_errors.csv"
        audit.parent.mkdir(parents=True)
        audit.write_text("dataset,run_id\n" + "arc,arc-bilstm-corn-s0\n" * 15, encoding="utf-8")
        self.assertFalse(self.make().tasks["F2"].done)             # the grid has not come back
        self.assertTrue(self.make(reported={"F2"}).tasks["F2"].done)  # a report always counts
        self.assertTrue(self.make(reported={"C5"}).tasks["F2"].done)  # found in the repository once the grid is back

    def test_everyone_owes_the_common_hand_offs(self):
        state = self.make(reported={"C:slides", "D:slides"})
        owing = [line[1] for line in state.waits_for("B") if line[2] == "slides"]
        self.assertEqual(owing, [["A", "E", "F"]])
        self.assertEqual(state.missing("slides"), ["A", "B", "E", "F"])
        self.assertFalse(state.arrived["slides"])
        self.assertTrue(state.blocked(state.tasks["B5"]))

    def test_a_hand_off_to_all_five_others_is_listed_with_its_source_only(self):
        state = self.make()
        self.assertIn("tables", [line[2] for line in state.waited_by("A")])
        self.assertNotIn("tables", [line[2] for line in state.waits_for("D")])
        self.assertEqual(state.waits_for("D"), [])

    def test_status_follows_the_next_task(self):
        state = self.make(reported={"F6", "F7"})
        self.assertEqual(state.status("F"), "waiting")  # F1 to F4 all need the grid
        self.assertEqual(state.status("C"), "free")     # C1 can be done; only C7 waits
        self.assertEqual(state.status("A"), "partly")   # the tables wait for the grid, the README does not
        self.assertEqual(state.status("D"), "free")
        everything = set(state.tasks)
        self.assertEqual(self.make(reported=everything).status("B"), "finished")
        self.assertIsNone(self.make(reported=everything).bottleneck())

    def test_both_languages_carry_the_same_facts(self):
        pulls = [{"number": 12, "url": "https://github.com/example/levellens/pull/12", "login": "someone"}]
        for now in (FRIDAY_EVENING, board.when(14, "13:00"), board.when(15, "16:00")):
            state = self.make(reported={"C1", "E:slides"}, now=now, pulls=pulls)
            english, chinese = state.render("en"), state.render("zh")
            self.assertEqual(facts(english), facts(chinese), now)
            self.assertEqual(english.count("](work/"), 6)
            self.assertEqual(chinese.count(".zh-CN.md)"), 6)

    def test_the_pages_change_only_when_the_board_changes(self):
        for page in board.PAGES.values():
            text = (self.folder / page).read_text(encoding="utf-8")
            if board.START not in text:
                text += f"\n{board.START}\n{board.END}\n"
            (self.folder / page).write_text(text, encoding="utf-8", newline="\n")
        state = self.make()
        boards = {lang: state.render(lang) for lang in board.PAGES}
        self.assertTrue(board.update(self.folder, boards, FRIDAY_EVENING))
        written = (self.folder / "README.md").read_text(encoding="utf-8")
        self.assertIn("Last change: Fri 13 Nov 21:00", written)
        self.assertIn("上次变化：11 月 13 日（周五）21:00", (self.folder / "README.zh-CN.md").read_text(encoding="utf-8"))
        self.assertFalse(board.update(self.folder, boards, board.when(13, "21:15")))  # only the clock moved
        self.assertEqual(written, (self.folder / "README.md").read_text(encoding="utf-8"))
        later = self.make(now=board.when(14, "10:30"))                                # C5 is now past its time
        self.assertTrue(board.update(self.folder, {lang: later.render(lang) for lang in board.PAGES}, board.when(14, "10:30")))


if __name__ == "__main__":
    unittest.main(verbosity=1)
