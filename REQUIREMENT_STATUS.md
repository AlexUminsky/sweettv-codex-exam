# Requirement matrix — actual evidence, WORKING

This matrix is a status navigation aid, not certification of grading or acceptance. The exam and guidelines govern.

| Requirement | Status and evidence |
|---|---|
| Native recording gate | PASS: `audit/recording_check.json` at handoff root; bound session 01a196eb-7cde-7743-a95b-23446e2f52fc; exact previous probe tool output and corresponding tool input manually inspected in native JSONL. Current recording continues; final copy pending. |
| Versions/current time | CLI 0.166.0, turn model gpt-6.1-sol; actual 2026-11-01 timestamps; `results/environment.json`, `requirements.txt`. |
| Authoritative conditions | Exam MD/checklist read; 12-page guidelines and 10-page midterm text extracted locally in audit/course_text; relevant class feedback read. Image-only course slides not claimed as text-read. |
| Provenance | Prior ChatGPT Work attempt and handoff disclosed in AI_USE.md and report. No previous solution imported. Instructor question about earlier assistance log remains unresolved. |
| ZIP CRC/safe extraction | PASS: exact five members, no paths, CRC `None`, sizes and hashes compared to extracted files; `audit/input_manifest.json`; reproducible prepare_data.py. |
| Kaggle description/threshold | Read full data-description/evaluation via public PageService/ListPages, saved `audit/kaggle_pages.json`; ≥80% full-airing union rule. |
| Correct overlap/fragments/ID0 | pipeline.py, positive/nested/multiple brute-force tests; union eliminates double count; actual log audit. Invalid rows excluded; ID strings preserved including leading zeros. |
| Temporal fitting | Explicit May18/Jul27 boundaries; full airing exclusion at crossing; training-only vocabulary/IDF; compact observed-item ALS; future schedule availability only. |
| Two required methods + baseline | Three independently implemented/evaluated: content, directional rules, implicit ALS; popularity and repeat separately. Eight configurations declared before initial build. |
| Fair comparison | Shared 6,376 candidate pool, all 1,260 template cohort, repeats allowed, popularity/lexical fallback and ties. |
| MAP/cohort/cold start | 6 synthetic tests PASS, independent AP all methods PASS; no-target AP0; 18 train-cold template, 22 no-target template, sensitivity on 1,238 active and 4,643 all-active. |
| Selection/refit/CSV | Repeat selected; nine outputs including one separately declared diagnostic; full refit through Jul26, 1,260 rows, 5 distinct available nonzero IDs; independent every-row recomputation. CSV SHA-256 de641c8068e7d3f0bc93719a074a3b8ca1055ecb3b4836369e02635248af17e5. |
| Runtime/memory/seeds | Clean corrected build and original run separately logged; 37.62 s clean pipeline, actual RSS in environment.json, seed 20261101. |
| Logs/predictions/notebook | Per-method CSVs, initial results, executed four-cell analysis.ipynb, transparent analyze.py, tests/analysis/build logs. |
| Real failure/one-variable confirmation | Content 0.205927 -> 0.313923 after β0->.25 repeat addition, 727 improved/44 worsened; remains below repeat. ALS initialization review correction, original results retained; real analysis KeyError traceback and fix retained. |
| Same two users/examples | Heavy10500740270951698446, cold10120801182216576239 across all configurations; ordered ID tables plus exact-title/category dictionary; histories and held-out targets. |
| Interpretation | Offline habit/discovery/concentration/fallback comparison; no enjoyment/never-seen/revenue/A-B or measured PageRank claims. |
| Primary sources | Four opened sources and consulted sections in references.json and report. Unavailable Stanford/Agrawal pages are not cited. |
| Student AP arithmetic | RECEIVED AND CORRECT, exact student words in personal_checks.json and native session. |
| Student interval arithmetic | RECEIVED AND CORRECT, exact student words in personal_checks.json and native session. |
| Student design/limitation/reflection | RECEIVED: verbatim normalization/limitation explanation and explicit trust boundary, with English translations in personal_checks.json. Personal checks cover one AP@5 and one viewing-fraction example only. |
| English report structure/cap | WORKING PDF: five content pages plus one reference page, four-sentence abstract; email received. Six draft pages rendered and visually viewed; final evidence will require re-render/review. |
| Public bundle | Explicit allowlist github_upload, code/versions/results/notebook/identical CSV; no raw data/native transcript/secrets/env. |
| GitHub link and contents | Repository publicly readable at https://github.com/AlexUminsky/sweettv-codex-exam; initial checked commit 8b4181ba4cd556c9f681cc6c84f08e71942d7905 contains only 12 root files. REQUIRED FIX: upload src/, tests/, results/, logs/, audit/ from github_upload. Public CSV byte-identical to project and bundle. See audit/github_verification.json. |
| Genuine private score | Screenshot evidence/kaggle_private_score.png visually read: Complete (after deadline), private 0.39237, public 0.40278, submission.csv, student name/account visible. Student confirms upload without resaving. Numeric submission reference not visible: PENDING reference/link. Kaggle-uploaded bytes not independently downloaded; attestation is explicit. |
| Final PDF | Evidence incorporated into WORKING draft; PENDING missing public implementation and submission reference before final label. |
| Final native byte-copy | PENDING after final assistant turn and Codex exit; student runs collector externally. No copying a still-growing log as finished. |
| LMS JSONL/size acceptance | Native JSONL retained; if LMS rejects extension or >10 MB, student must ask instructor. No format exception promised. |

Do not submit the WORKING draft as a completed report. Before final completion refresh this matrix against the actual external evidence and final PDF.
