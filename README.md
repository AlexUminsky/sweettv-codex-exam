# SWEET.TV: a fresh recorded implementation

Alexander Uminsky · aleksandr.uminski@gmail.com · Individual · 2026-11-01

This work was built and run in one native Codex CLI session. ChatGPT Work made an earlier attempt and prepared the handoff, prompts, and recording tools. No previous implementation, predictions, or measurements were imported. The native recording is private and must not be uploaded to GitHub. See `AI_USE.md`.

## Reproduce

Use Python 3.13 (measured: 3.13.9, macOS arm64). This repository does not contain the raw competition data. Obtain the ZIP through your own Kaggle account after accepting the rules; put it in `data/sweettv-tv-program-recommender.zip` relative to the repository root. Commands below assume this README is at repository root, as in the prepared GitHub bundle.

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python src/prepare_data.py data/sweettv-tv-program-recommender.zip data/extracted
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python src/pipeline.py --data data/extracted
.venv/bin/python src/analyze.py data/extracted
.venv/bin/python src/make_notebook.py
```

In the local handoff workspace the code lives under `project/`; prefix `.venv`, `src/`, `tests/` and requirements paths with `project/`, while `data/` remains at workspace root. The notebook generator runs analysis on already produced results and creates a real executed notebook; it is a small sequential executor for these plain Python cells, with stdout captured and execution counts assigned after successful execution. It does not synthesize outputs. `analyze.py` is the equivalent transparent executable analysis.

Full pipeline rereads 5.8M log rows in chunks of 200,000, unions overlap fragments, fits eight predeclared configurations and selects on validation. About 2 GB peak RSS was observed; timings are in `results/`. `--reuse-overlap` is an optional local cache path and requires the private derived overlap file, deliberately excluded from the public bundle. Run without it for clean reproduction.

## Protocol

Read Kaggle's **Data** and **Evaluation** pages. The Data description specifies at least 80% screen time per show, then each user's five most frequent shows. The actual files spell the session-duration column `duraton`, unlike the prose description. We compute session length from timestamps, not from that column. Programme length comes from the schedule's duration in seconds. There were 117 duration mismatches in this run.

Use half-open same-channel intervals, all positive overlaps (including airings that began before a session, and concurrent broadcasts). Union all fragments per user and exact airing; qualify if union seconds >= 0.8 times full airing duration. Each qualified airing contributes one count. Invalid dates/IDs/channel or nonpositive durations are dropped. Blank metadata contributes no token. ID `0` is excluded, never treated as a dislike; all IDs stay exact strings, including leading zeros. Conflicting labelled airings can both match; identical `(channel,start,end,item)` rows are deduplicated. No inference of enjoyment or never-seen status is made.

Train: full airings within [2020-03-09, 2020-05-18). Validation: full airings within [2020-05-18, 2020-07-27). An airing crossing a split boundary is excluded, even if its viewing fragments lie on only one side. A session crossing a boundary is clipped to each eligible airing. Schedule file names are not split criteria: files spill beyond their nominal weeks. Future schedule rows define availability only. Final training ends 2020-07-27; candidate starts are in [2020-07-27,2020-10-19). All times are dataset wall times, without an assumed timezone conversion.

Vocabulary, IDF and item metadata come only from completed training airings. Future-only IDs have zero content vectors and zero collaborative factors. ALS is fit in a compact training-observed item space; future IDs cannot influence initialization. A future schedule cannot teach preferences or provide content features in this protocol.

Primary cohort is all 1,260 template users, not selected by validation activity; no-target AP=0. Secondary metrics cover 1,238 template users with a qualified validation view, and 4,643 active validation users overall. There are 18 template users with no qualifying training history, 22 with no validation targets, and four with no history after final refit. The template itself is a competition-provided selected population; these metrics do not estimate performance for all platform accounts. Local target sets may contain fewer than five IDs; their denominator is min(5, |targets|).

Candidates: every nonzero ID airing in the prediction window. Repeats are allowed everywhere. All methods rank positive scores, then training distinct-user popularity, then exact lexical ID. Nonpositive model scores are clipped to zero; zero-score slots and cold users receive the same popularity fallback. Exactly five distinct IDs are returned. Lexical ordering is an explicit deterministic tie choice, not numeric order; it preserves leading zeros.

MAP@5 is the macro mean of AP@5: sum of precision at first occurrences of relevant hits in the five ranked positions, divided by min(5, number of relevant targets). No-target AP=0. Ground truth is the five most frequent >=80% watched validation items, frequency ties lexical. Standard AP tests relevance membership; the ordering of predicted hits matters, while reordering the ground-truth set does not.

## Methods and selection

* Popularity: number of distinct training users per item.
* Repeat: qualified-airing counts per user and item.
* Content: categorical one-hot category, three genre fields, year string, director and comma-separated actor tokens; training-item IDF, L2 item vectors, log-count-weighted normalized user profile, cosine score. Title is not a feature.
* Directional rules: one entire training user history is a basket; binary pair counts, confidence `n_ij/(n_i+shrink)`, minimum co-view user count 2 or 5, no self-rules, strongest 100 consequents per antecedent. Aggregate with L1-normalized log-count history. It models co-occurrence, not temporal causation.
* Implicit ALS: binary preference, confidence `1+alpha*log1p(count)`, unobserved confidence 1, regularization 1, eight alternating iterations, factors 16 or 32, alpha 10 or 40, seed 20261101. This is implemented directly with NumPy/SciPy, not an external recommender package.

`search_plan.json` was written before running the initial eight configurations. All components were evaluated individually. An ALS initialization review correction was then applied; `results/initial_run/` retains the first results and `results/metrics.csv` contains the corrected ones. `diagnostic_plan.json` declared one content-plus-repeat ablation after observing content's failure; it adds exactly `0.25*log1p(count)/max_user_logcount` to the content score. No private score was used to choose anything. This diagnostic is reported separately. Repeat remained the selected strategy: 0.355864 MAP@5 versus 0.313923 for this diagnostic; the actual selection evidence is in `results/`.

## Artifacts

`submission.csv` is the designated CSV. `results/metrics.csv`, `results/content_ablation.json`, `results/two_users.json`, per-method validation predictions, independent checks, selected configuration, input/output hashes, logs and `analysis.ipynb` provide evidence. `requirements.txt` records resolved versions. Synthetic checks appear only in test logs, not the experiment table. Raw logs, extracted data, derived per-airing overlaps, environments, credentials, and native session are excluded from the GitHub bundle.

See `MANUAL_UPLOAD_RU.md` for browser-only upload steps. The report remains WORKING until the genuine external evidence and personal checks are received.
