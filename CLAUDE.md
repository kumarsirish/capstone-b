# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

This is a capstone learning package for **Sahayak Health AI** (सहायक, "aid"), an AI triage decision-support agent for India's frontline health workers (ASHA/ANM). It is built on Google ADK (`LlmAgent` / `SequentialAgent`) with either Gemini (`GOOGLE_API_KEY`) or a local Ollama model (`hermes3:8b`) via LiteLLM. The agent reads a symptom description and recommends exactly one of three care levels — `WAIT`, `DOCTOR`, `ER` — never a diagnosis or prescription.

All actual project code lives under `starter-files/`; treat that as the effective package root. `dataset/` at the repo root is a cached copy of the `gretelai/symptom_to_diagnosis` HF dataset, and `capstone-B-workflow.pdf` is a reference doc.

## Folder structure (`starter-files/`)

- `learner/` — the editable/gradable work: four notebooks plus `sahayak_starter.py` (agent instructions, policy baseline, safety evaluator). This is what gets submitted.
- `src/` — provided support modules (`demo_app.py`, `eval_agent.py`, `setup_db.py`, `data_loader.py`, `sahayak_tools.py`, `utils.py`). Not submitted; imports the learner's `sahayak_starter.py` from `../learner` by inserting it onto `sys.path`.
- `data/` — `cases.db` (SQLite + embeddings) and the raw `medical_triage_500.jsonl` dataset.
- `tests/` — pytest harness (`test_sahayak_harness.py`) that checks the safety evaluator, escalation floor, and follow-up relevance logic without calling any LLM/ADK.
- `docs/` — `CAPSTONE_PROBLEM_STATEMENT.md` (architecture + grading philosophy) and `LEARNER_TASK_BRIEF.md` (deliverables and marks per notebook).

Both `learner/` and `src/` keep their own copies of shared modules (`utils.py`, `data_loader.py`, `sahayak_tools.py`) so each notebook folder can run standalone; `tests/` (via `conftest.py`) imports the canonical copies from `src/` plus the learner's `sahayak_starter.py` from `learner/`.

## Common commands

Run all commands from `starter-files/`.

```bash
pip install -r requirements.txt

# one-time: build cases.db (RAG case-memory DB) if not already present
cd src && python setup_db.py

# run the safety-harness test suite (12 tests, no LLM calls)
pytest tests/
pytest tests/test_sahayak_harness.py::test_red_flag_case_escalates_to_er   # single test

# interactive demo app (FastAPI) — serves / (single query) and /dashboard (trust dashboard)
cd src && python demo_app.py     # http://localhost:7860

# live-agent evaluation against the held-out HF test split (calls the real LLM pipeline)
cd src && python eval_agent.py --n 50
cd src && python eval_agent.py --n 100 --seed 42

# notebooks (run from learner/, since they do sys.path.insert(0, '../src'))
cd learner && jupyter notebook adk_foundations.ipynb
```

Environment: copy `.env.example` to `.env` and set `GOOGLE_API_KEY` for Gemini, or run Ollama locally with `hermes3:8b` pulled (`ollama pull hermes3:8b`) — the notebooks and eval scripts auto-fall back to Ollama when no Gemini key is set.

## Architecture — two-phase interactive ADK pipeline

```
PHASE A — intake
patient_input -> symptom_parser -> severity_scorer -> followup_asker
    [PAUSE — ASHA worker answers the follow-up question]
PHASE B — decision
triage_decider (reads the answer, may escalate, never de-escalates)
    -> response_formatter -> final_response
    -> safety_evaluator (post-hoc audit, not part of the decision path)
```

Batch/unattended evaluation runs the same pipeline with `auto=1`, passing `answer="(not provided)"`; the decider then falls back to the base severity rule. Each stage has exactly one job so failures can be attributed to a specific stage (extraction, severity, follow-up, triage, formatting, or audit).

**The judge never grades itself.** Reported safety metrics come from the deterministic rule-based `safety_evaluator_agent()` in `sahayak_starter.py` (regex + rule checks: disclaimer present, no diagnosis/prescription language, valid triage label, no red-flag under-triage vs. reference). The stage-6 LLM auditor is a teaching device only — its agreement with the deterministic judge is reported as a separate metric, never used as the actual pass/fail signal.

`triage_decider` keeps four tools available regardless of the decision path: vitals extraction, NEWS2 scoring, case-memory DB search (`search_symptom_cases_db()` against `cases.db`), and drug safety lookup. The case-memory DB is evidence shown to the agent, not the grading ground truth — grading ground truth is always `TRIAGE_MAP` in `data_loader.py`, applied to `gretelai/symptom_to_diagnosis` diagnosis labels.

Key safety invariant (`escalation_floor()` in `sahayak_starter.py`): a worrying follow-up answer can only escalate the triage level, never de-escalate it — enforced by an ambiguous-severity-band floor (severity 2 + red-flag answer -> at least `DOCTOR`; severity 3 + hard red flag -> `ER`).

Optional distinction-path architecture (not the core learner path): replace the single-threaded Phase A with a `ParallelAgent` of `red_flag_reviewer` / `vitals_reviewer` / `guideline_reviewer` feeding a `triage_synthesizer`. The final triage decision itself must stay single-threaded — one accountable decision node, not competing parallel answers.

Train/test split discipline: the **train** split feeds notebooks, the policy baseline, and DSPy dev sampling. The **test** split is reserved for `eval_agent.py` live-agent evaluation and must never be used for prompt tuning. The DSPy MIPROv2 optimization pass (in the evaluation-and-optimisation notebook) compiles the triage-decision prompt on 20 hand-authored train cases, evaluated before/after on a held-out dev set never shown to the optimizer, gated on a clinically weighted loss matrix (ER→WAIT=10, DOCTOR→WAIT=5, ER→DOCTOR=3, over-triage=1) written to `dspy_gate_results.json`.

## Grading-relevant constraints

These aren't generic best practices — they're the specific rules the evaluator and grader check against, so treat them as correctness requirements when touching `sahayak_starter.py`, prompts, or the evaluators:

- Final response must give exactly one of `WAIT` / `DOCTOR` / `ER`, include the required disclaimer verbatim ("This is decision support guidance only. Always consult a qualified medical professional for diagnosis and treatment."), use India-specific emergency guidance (nearest hospital/CHC/PHC or call 108), and avoid diagnosis/prescription language.
- Acceptance thresholds referenced by the eval harness: under-triage rate ≤ 5% (ACS field-triage standard) and ER-sent-home = 0, over-triage ≤ 50%, accuracy ≥ 60%, follow-up policy compliance ≥ 90% (ask exactly when severity is 2–3), follow-up relevance ≥ 90%, loop-closure escalation ≥ 80%, label-fallback rate ≤ 2%.
- Severity rubric calibration lesson: symptom drama is not urgency (e.g. "severe headache with vomiting" is a classic migraine WAIT pattern, not an escalation trigger) — rubric changes should be validated against the dataset's actual condition archetypes, not surface intensity language.
