"""The live status board on the home page: who has done what, and who waits for whom.

    python .github/scripts/board.py             # rewrite the board in README.md and README.zh-CN.md
    python .github/scripts/board.py --print     # print the English board and change nothing
    python .github/scripts/board.py --offline --print   # the same without asking GitHub: no reports, no pull requests

Where the board gets its facts:

1. The repository. A task that leaves a file or result rows in `main` is found without a report.
2. Reports. A comment on the issue of a share with a line such as `done C4`, `done slides` or `undo C4`.
3. The open pull requests.

The workflow `.github/workflows/board.yml` runs this file after every push, comment and pull request,
and every 15 minutes. The two pages are committed only when the board has changed.

The tasks, their short names and the owners are read from `work/README.md` and `work/README.zh-CN.md`.
This file adds what those pages do not say in a form a program can read: when a task is due, how to
see that it is done, and which hand-off it needs first.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SGT = timezone(timedelta(hours=8))  # every time on the pages is Singapore time
PAGES = {"en": "README.md", "zh": "README.zh-CN.md"}
OVERVIEW = {"en": "work/README.md", "zh": "work/README.zh-CN.md"}
WORKFLOW = ".github/workflows/board.yml"
START, END = "<!-- board:start -->", "<!-- board:end -->"
STAMP = "{stamp}"  # stands for the time of the last change until the page is written
SHARES = "ABCDEF"
ISSUE = {share: number for number, share in enumerate(SHARES, start=1)}  # share A is issue 1
DATASETS = ("race", "arc", "math")
TRANSFORMER_GRID = "configs/grid/tier1_transformers.txt"
BILSTM_GRID = "configs/grid/tier1_bilstm.txt"
LINES = 3  # lines in one cell before "+ n more"


def when(day: int, clock: str) -> datetime:
    """Return a time in November 2026, Singapore time: when(14, "10:00")."""
    hour, minute = clock.split(":")
    return datetime(2026, 11, day, int(hour), int(minute), tzinfo=SGT)


# ---- the plan, in a form a program can read -------------------------------------------------
#
# A check says how the repository shows that a task is done:
#   ("done",)                        finished before the board existed
#   ("report",)                      nothing in the repository shows it: the owner reports it
#   ("files", [paths])               all these files exist
#   ("rows", [models], [datasets])   every model has a result row on every dataset (default: all three)
#   ("variant", text, [datasets])    a result row whose variant holds this text, on every dataset
#   ("grid", [grid files])           every line of these grid files has a result row
#   ("final_files", glob, datasets)  a file per dataset, named <dataset>-..., none from a smoke run
#   ("final_csv", path, rows)        the table has at least this many lines and none from a smoke run
#   ("all", check, check, ...)       every one of these checks
#
# A task found in the repository counts only when the hand-offs it needs have arrived. Outputs made
# from models that are not final therefore do not close a task. A report always counts.

# task id: (due, check, hand-offs it needs)
NUMBERED = {
    "A1": (when(12, "16:00"), ("done",), ()),
    "A2": (when(12, "16:00"), ("done",), ()),
    "A3": (when(12, "16:00"), ("done",), ()),
    "A4": (when(12, "16:00"), ("done",), ()),
    "A5": (when(15, "14:00"), ("report",), ()),
    "A6": (when(14, "22:00"), ("report",), ()),
    "A7": (when(15, "11:00"), ("report",), ("recording",)),
    "A8": (when(15, "15:00"), ("report",), ("recording",)),
    "B1": (when(13, "20:00"), ("files", ["docs/datasets.md"]), ()),
    "B2": (when(13, "23:59"), ("files", ["docs/related_work.md"]), ()),
    "B3": (when(14, "15:00"), ("report",), ("errors",)),
    "B4": (when(14, "10:00"), ("report",), ()),
    "B5": (when(14, "18:00"), ("report",), ("slides",)),
    "B6": (when(15, "14:00"), ("report",), ("reflection",)),
    "B7": (when(15, "15:00"), ("report",), ()),
    "C1": (when(12, "23:59"), ("files", ["docs/compute.md"]), ()),
    "C2": (when(13, "10:00"), ("report",), ()),
    "C3": (when(13, "12:00"), ("report",), ()),
    "C4": (when(13, "23:59"), ("grid", [TRANSFORMER_GRID]), ("configs",)),
    "C5": (when(14, "10:00"), ("grid", [TRANSFORMER_GRID]), ()),
    "C6": (when(14, "12:00"), ("report",), ()),
    "C7": (when(15, "10:00"), ("report",), ("readme", "code")),
    "D1": (when(12, "23:59"), ("all", ("rows", ["majority", "stratified_random"]), ("rows", ["subject_prior"], ["math"])), ()),
    "D2": (when(13, "15:00"), ("rows", ["tfidf_logreg", "tfidf_svm", "tfidf_cnb"]), ()),
    "D3": (when(13, "15:00"), ("rows", ["r2de"]), ()),
    "D4": (when(13, "20:00"), ("rows", ["linguistic_lightgbm"]), ()),
    "D5": (when(14, "10:00"), ("files", [f"results/figures/share_d_shap/{name}_global_shap.png" for name in DATASETS]), ()),
    "D6": (when(13, "23:59"), ("rows", ["qwen_zero_shot"]), ()),
    "E1": (when(12, "23:59"), ("files", ["levellens/heads/ce.py", "levellens/heads/corn.py"]), ()),
    "E2": (when(13, "20:00"), ("grid", [BILSTM_GRID]), ()),
    "E5": (when(13, "12:00"), ("files", ["levellens/models/transformer.py"]), ()),
    "E6": (when(13, "12:00"), ("files", [TRANSFORMER_GRID]), ()),
    "E7": (when(13, "23:59"), ("variant", "optuna", ["arc"]), ()),
    "E8": (when(14, "10:00"), ("final_files", "results/export/*.json", DATASETS), ("results",)),
    "F1": (when(14, "12:00"), ("final_files", "results/explain/*_cue_words.csv", DATASETS), ("results", "models")),
    "F2": (when(14, "12:00"), ("final_csv", "results/audit/worst_errors.csv", 15), ("results",)),
    "F3": (when(14, "12:00"), ("final_csv", "results/transfer/transfer.csv", 9), ("results", "models")),
    "F4": (when(14, "12:00"), ("final_csv", "results/transfer/extrapolation_summary.csv", 1), ("results", "models")),
    "F6": (when(14, "14:00"), ("files", ["app/streamlit_app.py"]), ()),
    "F7": (when(14, "14:00"), ("files", ["app/pages/results.py"]), ()),
}

# Work that has no task number on the share pages. Reported by its word: `done tables`.
# word: (share, due, check, hand-offs it needs, English name, Chinese name)
EXTRA = {
    "tables": ("A", when(14, "12:00"), ("report",), ("results",),
               "Final tables and figures, pass or fail for R1–R8", "表格和图定稿，R1–R8 逐条判定"),
    "code": ("D", when(14, "22:00"), ("files", ["levellens/models/classic.py"]), (),
             "Code behind the baseline rows", "基线结果背后的代码"),
}

# The tasks everyone has. Reported by their word: `done slides`.
# word: (due, English name, Chinese name)
COMMON = {
    "slides": (when(14, "16:00"), "Own slides, final", "自己的 slides 定稿"),
    "recording": (when(14, "23:59"), "Own recording", "自己的录制"),
    "reflection": (when(15, "10:00"), "Own lines for the reflection document", "反思文档里自己的内容"),
}

# Who hands what to whom, and by when. A hand-off has arrived when its task is done.
# `source` "*" means everyone hands it over, each when their own common task is done.
# name: (source, receivers, task, due, English, Chinese)
HANDOFFS = {
    "configs": ("E", "C", "E6", when(13, "12:00"), "Tier 1 grid configs", "Tier 1 的实验网格配置"),
    "results": ("C", "AEF", "C5", when(14, "10:00"), "result rows of the Tier 1 grid", "Tier 1 网格的结果行"),
    "models": ("E", "F", "E8", when(14, "10:00"), "final models behind the Predictor API", "Predictor API 后面的最终模型"),
    "errors": ("F", "B", "F2", when(14, "10:00"), "list of the 15 worst errors", "最严重的 15 个错误的清单"),
    "tables": ("A", "BCDEF", "A:tables", when(14, "12:00"), "final tables and figures", "定稿的表格和图"),
    "code": ("D", "AC", "D:code", when(14, "22:00"), "code behind the baseline rows", "基线结果背后的代码"),
    "readme": ("A", "C", "A6", when(14, "22:00"), "README and driver notebook", "README 和主 notebook"),
    "slides": ("*", "B", "slides", when(14, "16:00"), "final slides for proofreading", "slides 定稿，用于校对"),
    "recording": ("*", "A", "recording", when(14, "23:59"), "own recording", "自己的录制文件"),
    "reflection": ("*", "B", "reflection", when(15, "10:00"), "lines for the reflection document", "反思文档里自己的内容"),
    "video": ("A", "BCDEF", "A7", when(15, "11:00"), "joined video for the review", "拼好的视频，用于审看"),
}

TEXT = {
    "en": {
        "heading": "## Live status: who waits for whom",
        "intro": "A program rebuilds this table from the files in the repository and from the reports in the six share issues. Last change: {stamp}.",
        "bottleneck": "**Most people wait for:** {who} · {what} · by {due}. Waiting, directly or through someone else: {names}.",
        "no_bottleneck": "**Nobody is waiting for anybody.**",
        "columns": ["Who", "Can do now", "Waits for", "Who waits for this share"], "count": "{done} / {all} own tasks done",
        "common": ["Everyone hands in", "To", "By", "Still missing from"], "all_in": "✅ nobody",
        "finished": "✅ Finished", "free": "🟢 Can work", "partly": "🟡 Partly waiting", "waiting": "🔴 Waiting",
        "overdue": "⏰ {n} past the time",
        "nothing": "—", "nobody": "nobody", "everyone": "everyone", "join": ", ",
        "more": "+ {n} more", "by": "by {due}", "for": "for {ids}", "runs": " ({have} / {want} runs)",
        "line": "**{names}**: {what} · {due}",
        "pulls": "Pull requests waiting for the lead's review: {list}.", "none": "none", "pull": "[#{number}]({url}) by {login}",
        "legend": [
            "**Finished something?** Comment `done C4` on the issue of your share: {issues}. Several at once: `done C1 C2`. What everyone hands in: `done slides`, `done recording`, `done reflection`. To take one back: `undo C4`.",
            "🟢 the next task can be done now. 🟡 the next task waits for someone, a later one can be done. 🔴 every open task waits for someone.",
            "⏰ marks what is past its planned time and is neither found in `main` nor reported.",
            "Files and result rows in `main` are found without a report.",
        ],
        "automatic": " The table is rebuilt after every push and every comment, and every 15 minutes.",
    },
    "zh": {
        "heading": "## 实时状态：谁在等谁",
        "intro": "这张表由程序根据仓库里的文件和六个分工 issue 里的报告自动重建。上次变化：{stamp}。",
        "bottleneck": "**现在最多人在等：**{who} · {what} · {due} 前。直接或间接在等的人：{names}。",
        "no_bottleneck": "**现在没有人在等别人。**",
        "columns": ["谁", "现在能做", "在等谁", "谁在等这一份"], "count": "自己的任务完成 {done} / {all}",
        "common": ["每个人都要交", "交给谁", "什么时候", "还没交的人"], "all_in": "✅ 没有",
        "finished": "✅ 全部完成", "free": "🟢 可以做", "partly": "🟡 部分在等", "waiting": "🔴 在等别人",
        "overdue": "⏰ {n} 项已过时间",
        "nothing": "—", "nobody": "没有", "everyone": "所有人", "join": "、",
        "more": "另有 {n} 项", "by": "{due} 前", "for": "用于 {ids}", "runs": "（{have} / {want} 次运行）",
        "line": "**{names}**：{what} · {due}",
        "pulls": "等组长审阅的 pull request：{list}。", "none": "没有", "pull": "[#{number}]({url})，提交人 {login}",
        "legend": [
            "**做完了一项？**在自己那一份的 issue 下评论 `done C4`：{issues}。一次报几项：`done C1 C2`。每个人都要交的三样：`done slides`、`done recording`、`done reflection`。报错了要撤回：`undo C4`。",
            "🟢 下一项任务现在就能做。🟡 下一项任务在等别人，后面有能做的。🔴 所有没做完的任务都在等别人。",
            "⏰ 表示已经过了计划时间，而且在 `main` 里找不到，也没有人报告。",
            "进了 `main` 的文件和结果行不用报告，程序自己能找到。",
        ],
        "automatic": "每次推送、每条评论之后，以及每 15 分钟，这张表都会重建一次。",
    },
}
WEEKDAY_ZH = "一二三四五六日"
STAMP_PATTERN = {
    "en": re.compile(r"(Last change: )[A-Z][a-z]{2} \d{1,2} [A-Z][a-z]{2} \d\d:\d\d"),
    "zh": re.compile(r"(上次变化：)\d+ 月 \d+ 日（周.）\d\d:\d\d"),
}


def show_time(moment: datetime, lang: str) -> str:
    """Write a time as the pages do: "Sat 14 Nov 10:00", or "11 月 14 日（周六）10:00"."""
    moment = moment.astimezone(SGT)
    if lang == "zh":
        return f"{moment.month} 月 {moment.day} 日（周{WEEKDAY_ZH[moment.weekday()]}）{moment:%H:%M}"
    return f"{moment:%a} {moment.day} {moment:%b} {moment:%H:%M}"


# ---- what the repository shows --------------------------------------------------------------


def grid_run_ids(text: str) -> list[str]:
    """Return the run id of every line of a grid file, built as `levellens.run` builds it."""
    ids = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        settings = dict(token.split("=", 1) for token in line.split() if "=" in token)
        parts = [settings["data"], settings["model"], settings["head"], settings.get("variant", "")]
        if settings.get("limit_train"):
            parts.append(f"n{settings['limit_train']}")
        ids.append("-".join(part for part in [*parts, f"s{settings.get('seed', '0')}"] if part))
    return ids


class Repository:
    """The files of the repository and the rows of the results store."""

    def __init__(self, root: Path):
        self.root = Path(root)
        rows = {}
        store = self.root / "results" / "runs.jsonl"
        for line in store.read_text(encoding="utf-8").splitlines() if store.exists() else []:
            if line.strip():
                row = json.loads(line)
                rows[row["run_id"]] = row  # the newest row of a run id counts
        final = [row for row in rows.values() if "smoke" not in row.get("variant", "")]
        self.run_ids = {row["run_id"] for row in final}
        self.trained = {(row["dataset"], row["model"]) for row in final}
        self.variants = {(row["dataset"], row.get("variant", "")) for row in final}

    def read(self, path: str) -> str:
        file = self.root / path
        return file.read_text(encoding="utf-8") if file.exists() else ""

    def check(self, rule: tuple) -> tuple[bool, tuple[int, int] | None]:
        """Apply one check. Return (passed, (have, want)); the count is None when the check does not count."""
        kind, *args = rule
        if kind == "done":
            return True, None
        if kind == "report":
            return False, None
        if kind == "files":
            return all((self.root / path).exists() for path in args[0]), None
        if kind == "rows":
            datasets = args[1] if len(args) > 1 else DATASETS
            return all((dataset, model) in self.trained for dataset in datasets for model in args[0]), None
        if kind == "variant":
            return all(any(d == dataset and args[0] in variant for d, variant in self.variants) for dataset in args[1]), None
        if kind == "grid":
            ids = [run_id for path in args[0] for run_id in grid_run_ids(self.read(path))]
            have = sum(run_id in self.run_ids for run_id in ids)
            return bool(ids) and have == len(ids), (have, len(ids))
        if kind == "final_files":
            names = [file.name for file in self.root.glob(args[0]) if "smoke" not in file.name]
            return {name.split("-")[0] for name in names} >= set(args[1]), None
        if kind == "final_csv":
            lines = [line for line in self.read(args[0]).splitlines()[1:] if line.strip()]
            return len(lines) >= args[1] and not any("smoke" in line for line in lines), None
        if kind == "all":
            return all(self.check(part)[0] for part in args), None
        raise ValueError(f"Unknown check {kind!r}")


def overview(root: Path, lang: str) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Read one overview page. Return the owner of each share, its page, and the short name of each task."""
    text = (Path(root) / OVERVIEW[lang]).read_text(encoding="utf-8")
    owners, pages, titles = {}, {}, {}
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        share = re.fullmatch(r"\[([A-F])\]\(([^)]+)\)", cells[0])
        if share and len(cells) == 7:  # the owners table
            owners[share.group(1)] = re.sub(r"\s*[(（][^)）]*[)）]", "", cells[2]).strip()
            pages[share.group(1)] = "work/" + share.group(2)
        elif re.fullmatch(r"[A-F]\d{1,2}", cells[0]) and len(cells) == 3:  # a Tier 1 task table
            titles[cells[0]] = cells[1]
    return owners, pages, titles


# ---- what people report ---------------------------------------------------------------------

REPORT = re.compile(r"^\s*(done|undo)\b[\s:：]*(.*)$", re.IGNORECASE)


def report_key(word: str, share: str, lead: bool) -> str | None:
    """Return the task a reported word stands for, or None when this author may not report it here."""
    word = word.strip("`*.。")
    if re.fullmatch(r"[A-Fa-f]\d{1,2}", word):
        key = word.upper()
        return key if key in NUMBERED and (lead or key[0] == share) else None
    word = word.lower()
    if word in COMMON or (word in EXTRA and EXTRA[word][0] == share):
        return f"{share}:{word}"
    return None


def read_reports(comments: dict[str, list[dict]]) -> set[str]:
    """Return the tasks reported as done.

    `comments` maps a share to the comments on its issue: {"body", "association", "created"}.
    Anyone may report a task of the share whose issue the comment is on. The owner of the repository
    may report any numbered task anywhere. A line counts only if every word after `done` or `undo`
    is a task, so a sentence that happens to start with "done" changes nothing. The latest report wins.
    """
    events = []
    for share, items in comments.items():
        for item in items:
            lead = item.get("association") == "OWNER"
            for line in (item.get("body") or "").splitlines():
                match = REPORT.match(line)
                words = [word for word in re.split(r"[\s,，、;；]+", match.group(2)) if word] if match else []
                keys = [report_key(word, share, lead) for word in words]
                if keys and all(keys):
                    events += [(item["created"], match.group(1).lower(), key) for key in keys]
    reported = set()
    for _, verb, key in sorted(events, key=lambda event: event[0]):
        (reported.add if verb == "done" else reported.discard)(key)
    return reported


def repository_name() -> str:
    """Return "owner/name": from the workflow, or from the address of `origin`."""
    name = os.environ.get("GITHUB_REPOSITORY")
    if not name:
        url = subprocess.run(["git", "config", "--get", "remote.origin.url"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        name = re.search(r"github\.com[:/](.+?)(?:\.git)?\s*$", url).group(1)
    return name


def ask_github(path: str) -> list[dict]:
    """Return every item of one list of the GitHub API, page by page."""
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    items, page = [], 1
    while True:
        joiner = "&" if "?" in path else "?"
        request = urllib.request.Request(f"https://api.github.com/repos/{repository_name()}/{path}{joiner}per_page=100&page={page}")
        request.add_header("Accept", "application/vnd.github+json")
        if token:
            request.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(request, timeout=30) as response:
            batch = json.load(response)
        items += batch
        if len(batch) < 100:
            return items
        page += 1


def fetch() -> tuple[set[str], list[dict]]:
    """Ask GitHub for the reports in the six share issues and for the open pull requests."""
    comments = {
        share: [
            {"body": item["body"], "association": item["author_association"], "created": item["created_at"]}
            for item in ask_github(f"issues/{ISSUE[share]}/comments")
        ]
        for share in SHARES
    }
    pulls = [
        {"number": item["number"], "url": item["html_url"], "login": item["user"]["login"]}
        for item in ask_github("pulls?state=open")
    ]
    return read_reports(comments), pulls


# ---- the board ------------------------------------------------------------------------------


@dataclass
class Task:
    key: str        # "C4" for a numbered task, "A:tables" or "C:slides" for the others
    share: str
    due: datetime
    rule: tuple
    needs: tuple
    name: dict = field(default_factory=dict)  # language -> short name
    done: bool = False
    count: tuple[int, int] | None = None


class Board:
    """The state of every task and hand-off at one moment."""

    def __init__(self, repository: Repository, reported: set[str], pulls: list[dict] | None, now: datetime, root: Path = ROOT):
        self.now, self.pulls, self.root = now, pulls, Path(root)
        self.owners, self.pages, self.titles = {}, {}, {}
        for lang in PAGES:
            self.owners[lang], self.pages[lang], self.titles[lang] = overview(root, lang)
        self.tasks: dict[str, Task] = {}
        for key, (due, rule, needs) in NUMBERED.items():
            self.tasks[key] = Task(key, key[0], due, rule, needs, {lang: self.titles[lang][key] for lang in PAGES})
        for word, (share, due, rule, needs, en, zh) in EXTRA.items():
            self.tasks[f"{share}:{word}"] = Task(f"{share}:{word}", share, due, rule, needs, {"en": en, "zh": zh})
        for word, (due, en, zh) in COMMON.items():
            for share in SHARES:
                self.tasks[f"{share}:{word}"] = Task(f"{share}:{word}", share, due, ("report",), (), {"en": en, "zh": zh})
        found = {key: repository.check(task.rule) for key, task in self.tasks.items()}
        settled: dict[str, bool] = {}

        def done(key: str) -> bool:
            if key not in settled:
                task = self.tasks[key]
                settled[key] = key in reported or (found[key][0] and all(arrived(name) for name in task.needs))
            return settled[key]

        def arrived(name: str) -> bool:
            source, receiver, task = HANDOFFS[name][:3]
            if source == "*":
                return all(done(f"{share}:{task}") for share in SHARES if share != receiver)
            return done(task)

        for key, task in self.tasks.items():
            task.done, task.count = done(key), found[key][1]
        self.arrived = {name: arrived(name) for name in HANDOFFS}

    # -- questions about one share

    def own_tasks(self, share: str) -> list[Task]:
        """Return the tasks of a share, earliest first, without the three that everyone has."""
        mine = [task for task in self.tasks.values() if task.share == share and task.key.split(":")[-1] not in COMMON]
        return sorted(mine, key=lambda task: (task.due, task.key))

    def open_tasks(self, share: str) -> list[Task]:
        return [task for task in self.own_tasks(share) if not task.done]

    def blocked(self, task: Task) -> bool:
        return any(not self.arrived[name] for name in task.needs)

    def late(self, due: datetime) -> bool:
        return self.now > due

    def missing(self, word: str) -> list[str]:
        """Return the shares that have not handed in one of the things everyone hands in."""
        return [share for share in SHARES if not self.tasks[f"{share}:{word}"].done]

    def waits_for(self, share: str) -> list[tuple[datetime, list[str], str]]:
        """Return the open hand-offs this share receives: (due, the shares that owe it, hand-off name).

        A hand-off that goes to all five others is listed with its source only, not five times more.
        """
        lines = []
        for name, (source, receivers, task, due, _, _) in HANDOFFS.items():
            if source == "*":
                owing = [other for other in self.missing(task) if other != receivers]
                if share == receivers and owing:
                    lines.append((due, owing, name))
            elif share in receivers and len(receivers) < len(SHARES) - 1 and not self.tasks[task].done:
                lines.append((due, [source], name))
        return sorted(lines, key=lambda line: (line[0], line[2]))

    def waited_by(self, share: str) -> list[tuple[datetime, list[str], str]]:
        """Return the open hand-offs this share owes: (due, the shares that wait, hand-off name).

        What everyone hands in is not listed here. It has a table of its own.
        """
        lines = []
        for name, (source, receivers, task, due, _, _) in HANDOFFS.items():
            if source == share and not self.tasks[task].done:
                lines.append((due, list(receivers), name))
        return sorted(lines, key=lambda line: (line[0], line[2]))

    def bottleneck(self) -> tuple[str, set[str]] | None:
        """Return the open hand-off that holds up the most people, and those people.

        Only a hand-off whose own task is not waiting for anything can be the bottleneck. Someone
        waits for it directly, or waits for a hand-off whose task needs it.
        """
        def waiting(name: str, seen: frozenset = frozenset()) -> set[str]:
            source, receivers = HANDOFFS[name][:2]
            people = set(receivers)
            for other, (other_source, _, other_task, *_) in HANDOFFS.items():
                if other_source != "*" and other not in seen and name in self.tasks[other_task].needs and not self.arrived[other]:
                    people |= waiting(other, seen | {name})
            return people - {source}

        best = None
        for name, (source, _, task, due, _, _) in HANDOFFS.items():
            if source == "*" or self.arrived[name] or self.blocked(self.tasks[task]):
                continue
            people = waiting(name)
            rank = (-len(people), due)
            if best is None or rank < best[0]:
                best = (rank, name, people)
        return (best[1], best[2]) if best else None

    # -- the page

    def names(self, shares, lang: str, of: str | None = None) -> str:
        """Write a list of shares as names. The five others of `of` together are "everyone"."""
        shares = [share for share in SHARES if share in shares]
        if of is not None and set(shares) == set(SHARES) - {of}:
            return TEXT[lang]["everyone"]
        return TEXT[lang]["join"].join(self.owners[lang][share] for share in shares)

    def task_line(self, task: Task, lang: str) -> str:
        t = TEXT[lang]
        label = task.key if ":" not in task.key else ""
        count = t["runs"].format(have=task.count[0], want=task.count[1]) if task.count else ""
        late = " ⏰" if self.late(task.due) else ""
        return f"{label} {task.name[lang]}{count} · {show_time(task.due, lang)}{late}".strip()

    def handoff_text(self, name: str, lang: str) -> str:
        """Return what a hand-off is, with its count when it has one: "result rows ... (0 / 20 runs)"."""
        source, _, task, _, en, zh = HANDOFFS[name]
        count = self.tasks[task].count if source != "*" else None
        what = {"en": en, "zh": zh}[lang]
        return what + (TEXT[lang]["runs"].format(have=count[0], want=count[1]) if count else "")

    def handoff_line(self, share: str, line: tuple, lang: str, receiving: bool) -> str:
        t = TEXT[lang]
        due, shares, name = line
        text = t["line"].format(names=self.names(shares, lang, of=share), what=self.handoff_text(name, lang),
                                due=t["by"].format(due=show_time(due, lang)))
        if self.late(due):
            text += " ⏰"
        needed = [task.key for task in self.open_tasks(share) if name in task.needs and ":" not in task.key]
        if receiving and needed:
            text += " · " + t["for"].format(ids=t["join"].join(needed))
        return text

    def cell(self, lines: list[str], lang: str, empty: str) -> str:
        if not lines:
            return empty
        shown = lines[:LINES]
        if len(lines) > LINES:
            shown.append(TEXT[lang]["more"].format(n=len(lines) - LINES))
        return "<br>".join(shown)

    def status(self, share: str) -> str:
        """Return "finished", "waiting", "partly" or "free" for a share.

        waiting: every open task waits for a hand-off. partly: the next task (the earliest one)
        waits, a later one can be done. free: the next task can be done now.
        """
        open_tasks = self.open_tasks(share)
        if not open_tasks:
            return "finished"
        if all(self.blocked(task) for task in open_tasks):
            return "waiting"
        return "partly" if self.blocked(open_tasks[0]) else "free"

    def row(self, share: str, lang: str) -> str:
        t = TEXT[lang]
        mine = self.own_tasks(share)
        open_tasks = self.open_tasks(share)
        free = [task for task in open_tasks if not self.blocked(task)]
        who = [
            f"**{self.owners[lang][share]}** · [{share}]({self.pages[lang][share]})",
            t[self.status(share)],
            t["count"].format(done=len(mine) - len(open_tasks), all=len(mine)),
        ]
        overdue = sum(self.late(task.due) for task in open_tasks)
        if overdue:
            who.append(t["overdue"].format(n=overdue))
        cells = [
            "<br>".join(who),
            self.cell([self.task_line(task, lang) for task in free], lang, t["nothing"]),
            self.cell([self.handoff_line(share, line, lang, receiving=True) for line in self.waits_for(share)], lang, t["nobody"]),
            self.cell([self.handoff_line(share, line, lang, receiving=False) for line in self.waited_by(share)], lang, t["nobody"]),
        ]
        return "| " + " | ".join(cells) + " |"

    def render(self, lang: str) -> str:
        """Return the board of one page. The time of the last change is left as STAMP."""
        t = TEXT[lang]
        out = [t["heading"], "", t["intro"], ""]
        worst = self.bottleneck()
        if worst:
            name, people = worst
            source, due = HANDOFFS[name][0], HANDOFFS[name][3]
            out.append(t["bottleneck"].format(who=self.owners[lang][source], what=self.handoff_text(name, lang),
                                              due=show_time(due, lang), names=self.names(people, lang, of=source)))
        else:
            out.append(t["no_bottleneck"])
        out += ["", "| " + " | ".join(t["columns"]) + " |", "|" + "---|" * len(t["columns"])]
        out += [self.row(share, lang) for share in SHARES]
        out += ["", "| " + " | ".join(t["common"]) + " |", "|" + "---|" * len(t["common"])]
        for name, (source, receiver, word, due, en, zh) in HANDOFFS.items():
            if source == "*":
                late = " ⏰" if self.late(due) and self.missing(word) else ""
                missing = t["everyone"] if len(self.missing(word)) == len(SHARES) else self.names(self.missing(word), lang) or t["all_in"]
                what = en[0].upper() + en[1:] if lang == "en" else zh
                out.append(f"| {what} | {self.owners[lang][receiver]} | {show_time(due, lang)}{late} | {missing} |")
        if self.pulls is not None:
            listed = t["join"].join(t["pull"].format(**pull) for pull in sorted(self.pulls, key=lambda pull: pull["number"]))
            out += ["", t["pulls"].format(list=listed or t["none"])]
        issues = t["join"].join(f"[{share}](https://github.com/{repository_name()}/issues/{ISSUE[share]})" for share in SHARES)
        legend = [line.format(issues=issues) if "{issues}" in line else line for line in t["legend"]]
        if (self.root / WORKFLOW).exists():
            legend[-1] += t["automatic"]
        out += ["", *[f"- {line}" for line in legend]]
        return "\n".join(out)


# ---- writing the pages ----------------------------------------------------------------------


def block_of(page: str) -> str:
    """Return the board that is in a page now: the text between the two markers."""
    return page[page.index(START) + len(START) : page.index(END)].strip("\n")


def with_block(page: str, block: str) -> str:
    """Return the page with a new board between the two markers."""
    return page[: page.index(START) + len(START)] + "\n" + block.strip("\n") + "\n" + page[page.index(END) :]


def update(root: Path, boards: dict[str, str], now: datetime) -> bool:
    """Write the boards into the pages if one of them has changed. Return whether anything was written."""
    pages = {lang: (Path(root) / PAGES[lang]).read_text(encoding="utf-8") for lang in PAGES}
    same = all(STAMP_PATTERN[lang].sub(r"\g<1>" + STAMP, block_of(pages[lang])) == boards[lang] for lang in PAGES)
    if same:
        return False
    for lang in PAGES:
        block = boards[lang].replace(STAMP, show_time(now, lang))
        (Path(root) / PAGES[lang]).write_text(with_block(pages[lang], block), encoding="utf-8", newline="\n")
    return True


def main(argv: list[str] | None = None) -> None:
    """Run the command line."""
    parser = argparse.ArgumentParser(description="Rebuild the status board on the home page")
    parser.add_argument("--offline", action="store_true", help="do not ask GitHub: no reports, no pull requests; use with --print")
    parser.add_argument("--print", action="store_true", help="print the English board and change nothing")
    parser.add_argument("--now", help="pretend it is this Singapore time, for example 2026-11-14T09:00")
    args = parser.parse_args(argv)
    if args.offline and not args.print:
        parser.error("--offline would write a board without the reports. Use it with --print.")

    now = datetime.fromisoformat(args.now).replace(tzinfo=SGT) if args.now else datetime.now(SGT)
    reported, pulls = (set(), None) if args.offline else fetch()
    board = Board(Repository(ROOT), reported, pulls, now)
    boards = {lang: board.render(lang) for lang in PAGES}
    if args.print:
        sys.stdout.reconfigure(encoding="utf-8")
        print(boards["en"].replace(STAMP, show_time(now, "en")))
        return
    print("board updated" if update(ROOT, boards, now) else "board unchanged")


if __name__ == "__main__":
    main()
