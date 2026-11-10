# How we work

## Access

- This repository is public, so everyone can read it without an account.
- To push code, send your GitHub username to the lead in the group chat. You get an invitation by email. Accept it.
- Until then, fork the repository and open a pull request from your fork.

## Branches and pull requests

- `main` always runs. Only the lead merges into `main`.
- One branch for each piece of work, named with your share letter: `d/tfidf-models`, `e/corn-head`.
- Open the pull request early and keep it small: one task or less. Write the task id in the title, for example "D2 TF-IDF models".
- Your buddy or the lead reviews it. The lead merges it.
- Merge `main` into your branch every morning.

## Files

- Each file has one owner. The layout in the [README](../README.md) shows who. If you need a change in someone else's file, ask the owner.
- No data, model weights, recordings or slides in this repository. Data and weights come from `scripts/download_data.py`.
- No personal details: no matriculation numbers, emails, phone numbers, signatures or photos. No access tokens.
- No hard-coded paths. Use the environment variables in [interfaces.md](interfaces.md).

## Runs and results

- Every run starts from a config and writes one line to `results/runs.jsonl` through `levellens.results`. Do not edit that file by hand.
- Seeds are fixed. The splits are saved once by the lead and loaded by every model. Never re-split.
- The test split is for final numbers. Choose settings on dev, or by cross-validation on train.
- GPU jobs go through share C. Hand over the config and the command. C queues the job and returns the rows.

## Code

- A docstring on every public function. Type hints where they help.
- Settings live in configs, not in the code.
- One small test for each module: shapes, value ranges, one known result.
- You may reuse open-source code if its licence allows it. Say where it came from in a comment at the top of the file, and add it to the acknowledgements in the README.
- You present your own part in the video, so make sure that you can explain every part of your code.

## Talking to each other

- Blocked for more than an hour? Say so in the group chat.
- Tick a task in your share's issue when it is done. Post results in the group chat as they come: a number or a figure is enough.
- If your share is running more than 3 hours over its estimate, tell the lead. The lead moves a task to a share that is under.
- The lead decides changes of plan and writes them in this repository the same day.
