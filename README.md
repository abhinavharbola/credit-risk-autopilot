# Continuous Credit Risk Governance Pipeline

A self-governing ML pipeline that simulates a credit risk classifier's production lifecycle: score live batches, wait for delayed labels, detect drift, retrain a challenger, gate it against the champion with a bootstrap significance test, promote or reject it, and roll back if the predecessor beats it on live data.

Built on free-tier infrastructure only: no paid APIs, no GPU, no local database server, and no human in the loop once scheduled.

## Preview

<p align="center">
  <img src="assets/ui.png" width="720" alt="Line chart of drift share per batch with retrain-triggered batches called out, plus a raw drift-check table">
  <br>
  <sub>Drift tab: per-batch drift share against the base training reference.</sub>
</p>

> Additional screenshots in [`assets/`](assets/), one per dashboard tab.

## What this is

A model does not stay good just because it was good at launch. This project simulates what happens after deployment: the world drifts, and the model must be watched, challenged and sometimes replaced, automatically and defensibly.

On every clock tick the pipeline:

1. Resets the `@production` alias in MLflow to the latest champion recorded in Postgres, then scores the current batch with it.
2. Checks the batch for drift against the base training pool (Evidently, two-sample K-S test per feature).
3. Releases ground-truth labels for the earlier batch whose delay has elapsed.
4. If drift crosses the threshold and the retrain cooldown has elapsed, trains a challenger on a capped base sample plus recent labeled live rows, excluding the batch it will be judged on, and compares it to the champion on that batch.
5. Promotes the challenger only if the bootstrap CI on the metric delta clears zero, storing the gated window's metric and drift fingerprint as a reference.
6. Screens the champion for degradation against that reference and rolls back only if the previous champion is still significantly better on the same live batch than challenger.

Every decision is written to an audit log, including drift checks, rejected challengers, rollback checks and alias reconciliations. The loop runs on a schedule via GitHub Actions.

## Architecture

```mermaid
flowchart TD
    clock[clock advance<br/>claim with FOR UPDATE SKIP LOCKED] --> reconcile[reconcile @production alias<br/>to latest champion in Postgres]
    reconcile --> score[score batch<br/>@production, cached model]
    score --> drift[drift check<br/>K-S vs base training pool]
    drift -->|no labels due yet| stop[tick ends]
    drift --> release[release due labels<br/>delay window elapsed]
    release -->|no drift or cooldown active| rollback
    release -->|drift and cooldown elapsed| retrain[retrain challenger<br/>base sample + recent labeled rows<br/>excluding the gate batch]
    retrain --> gate[gate on the excluded labeled batch<br/>bootstrap CI on the metric delta]
    gate -->|rejected| audit1[audit_log: gate_evaluation rejected]
    gate -->|promoted| promote[promote: DB rows first,<br/>alias moved last]
    promote --> skip[rollback check skipped<br/>logged as skipped]
    audit1 --> rollback[rollback check]
    rollback -->|reference stale| suppress[suppress this cycle]
    rollback -->|drop below screen threshold| noop[no action]
    rollback -->|drop flagged| paired[paired bootstrap on the live batch<br/>previous champion minus current]
    paired -->|previous significantly better| revert[revert @production alias]
    paired -->|not significantly better| noop
```

The first three ticks have no labels due and end after the drift check. Each tick runs in one database transaction, so a failed tick rolls back its claim and rows and is retried cleanly. MLflow alias moves cannot join that transaction, so they happen last, and every tick starts by resetting `@production` to the database's champion.

## Governance parameters

All values live in `config/gate_config.yaml`.

| Parameter | Value | Why |
|---|---|---|
| Primary metric | AUC-PR | Accuracy is near meaningless at a ~6.7% positive rate. |
| Significance | Bootstrap CI lower bound on the matched-batch delta above 0 (alpha 0.05, 2,000 resamples) | The deciding test. Promotion is exactly equivalent to passing it. |
| Tolerance band | 0.01 | First rejection reason: worse than champion by more than this. Implied by significance, kept for a specific audit message. |
| Dominance | challenger > champion | Second rejection reason: a tie is rejected. Implied by significance, kept for the same reason. |
| McNemar | diagnostic only, reliable at 15+ discordant pairs | Logged, never decides an outcome. |
| Drift test | K-S p-value < 0.05 per feature | Forced, with every feature declared numerical. Evidently's default changes test with reference size. |
| Drift share threshold | 0.3 | 3+ of 10 features flagging by chance is rare under K-S at alpha 0.05 (about 1% per undrifted batch, assuming independence). |
| Retrain cooldown | 3 batches | Drift is measured against the base pool, so once persistent drift starts it never clears. Without a cooldown, every tick would retrain. |
| Retrain data | 3,000 base rows + up to 6,000 most recent labeled rows | Both caps configurable. The gate batch is always excluded. |
| Rollback screen | drop of 0.03 AUC-PR versus the stored reference | Cheap trigger only. The reference is one upward-biased gated batch, so it never decides a rollback alone. |
| Rollback decision | previous champion beats current on the same live batch, bootstrap CI lower bound above 0 | Same rigor as promotion, identical rows for both models. |
| Staleness | drift share moved by more than 0.3, or more than 30% of features disagree on drifted or not | Compares which features drifted. Raw p-values are uniform noise when nothing has drifted. |
| Delayed labels | 3 batches | Nothing is scored against labels that would not yet exist. |

## Design decisions that took more than one attempt to get right

- **The gate rejects noise, not luck.** Raw AUC-PR comparisons on 200-row batches mistook random variation for improvement, so the bootstrap CI on the matched-batch delta must clear zero.
- **The gate batch is out of sample for both models.** The challenger was once trained on the batch it was scored on, and the bootstrap champion on a pool containing every stream batch. The stream is now carved out before the base pool, and each retrain excludes the batch under evaluation.
- **Tolerance and dominance are nested inside significance.** A CI lower bound above zero already implies both, so they remain only as ordered rejection reasons. A randomized test asserts promotion equals the significance result.
- **Rollback compares like with like.** A live metric against a stored single-batch reference spans different windows and carries winner's-curse bias, so the reference only screens. The decision is a paired bootstrap of current versus previous champion on the same live batch.
- **The drift test is pinned, not defaulted.** Above 1,000 reference rows Evidently switches to a Wasserstein distance normalized by standard deviation, which misses shifts in heavy-tailed columns like `DebtRatio`. K-S is forced with every feature declared numerical, so thresholds do not depend on data volume.
- **Staleness compares which features drifted, not p-values.** With no drift, two p-values differ by more than 0.3 about half the time per column, which would suppress nearly every rollback check.
- **The database decides who the champion is.** MLflow alias moves cannot join a transaction, so they run last and every tick reconciles the alias to the latest non-rolled-back champion row.
- **Claim the batch first, without blocking.** `FOR UPDATE SKIP LOCKED` lets a racing run return immediately instead of waiting on the winner.
- **A challenger must be able to adapt.** Training on the full base pool diluted a few hundred new labels, so the base pool is sampled down and recent labeled rows are capped.
- **No retrain without labels, no rollback check right after a promotion.** `drift_detected` is always recorded, but `retrain_triggered` also needs a labeled batch and an elapsed cooldown, and a promoting tick skips the rollback check and logs `skipped`, since its reference was just derived from the same window.
- **Split before fitting.** Holdout and stream are carved out first, and imputation medians use the base pool only.
- **Never store the answer with the features.** Only model input columns are stored per prediction.

## Data and drift simulation

[Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) (Kaggle competition dataset, 150,000 rows, ~6.7% positive rate) is split, stratified by class, into three disjoint parts:

| Part | Rows | Role |
|---|---|---|
| Holdout | 22,499 | Frozen, never drifted. Scored only to record `holdout_metrics` at promotion. |
| Base pool | 63,751 | Trains the bootstrap champion, supplies the retrain base sample, and is the drift reference. |
| Stream pool | 63,750 | Batched into 318 batches of 200 rows (remainder dropped) to simulate live traffic. |

A recession scenario is injected deterministically per batch index (parameters in `config/drift_params.yaml`):

- **Persistent drift** (batch 10 onward, never reverts): `DebtRatio` and `RevolvingUtilizationOfUnsecuredLines` shift and scale upward, `MonthlyIncome` shrinks.
- **Temporary concept drift** (batches 15 to 19 inclusive, then reverts; `end_batch: 20` is exclusive): delinquent borrowers' values in five columns blend toward the non-delinquent centroid of the same batch. A single-class batch is left unchanged.

## Persistence

Postgres (Neon), four tables:

- `pipeline_state`: single-row clock, optimistic concurrency via a version column.
- `predictions`: one row per scored prediction, label filled in on release.
- `champion_history`: N-hop promotion lineage with window metrics, frozen-holdout metrics, drift fingerprint, staleness flag and rollback markers.
- `audit_log`: every governance decision by event type: `clock_advance`, `drift_check`, `label_release`, `gate_evaluation`, `promotion`, `rollback_check`, `rollback`, `alias_reconciled`.

Models and registry: MLflow on DagsHub, alias-based (`@production`, `@challenger`), never the deprecated stage API. Data: DVC against a DagsHub-hosted S3-compatible remote.

Dashboard trend chart and lineage deltas use the frozen-holdout metric, the only value measured on the same rows for every champion. The per-champion window metric is shown separately as the reference window.

## Project structure

```
credit-risk-autopilot/
├── assets/                     # dashboard icon and README screenshots
│
├── config/
│   ├── drift_params.yaml       # recession scenario
│   └── gate_config.yaml        # metric, thresholds, caps, cooldown
│
├── data/
│   ├── raw/                    # cs-training.csv, placed manually
│   └── processed/              # pickles generated by run_demo_loop.py
│
├── src/
│   ├── data/                   # ingest, stratified splits, drift injection
│   ├── db/                     # connection, repository, schema.sql
│   ├── model/                  # feature constants, training
│   ├── gate/                   # evaluate.py: metrics, bootstrap delta CI, gate decision
│   ├── drift/                  # Evidently K-S wrapper, fingerprint, staleness
│   ├── orchestration/          # clock, pipeline tick, promote / rollback / reconcile
│   ├── serving/                # FastAPI app
│   ├── llm/                    # single decision-explanation call
│   └── utils/                  # config and paths, logging, aliased model cache
│
├── dashboard/
│   ├── app.py
│   ├── styles.css
│   └── views/                  # overview, lineage, drift, audit_log
│
├── scripts/
│   ├── advance_clock.py        # entrypoint for cron and manual runs
│   ├── run_demo_loop.py        # bootstrap, then 25 ticks, then summary
│   └── smoke_test_mlflow.py    # connectivity check
│
├── tests/                      # 105 tests, 11 files
│
├── .github/workflows/
│   ├── ci.yml                  # lint and test against a Postgres service
│   └── cron_advance.yml        # scheduled clock advance
│
├── .streamlit/config.toml
├── .gitignore
├── pytest.ini
├── ruff.toml
├── requirements.txt
├── .env.example
└── README.md
```

## Getting started

### 1. Accounts (all free tier)

- [Neon](https://neon.tech) for Postgres.
- [DagsHub](https://dagshub.com) for MLflow tracking and registry, and a DVC-compatible remote.
- [Kaggle](https://www.kaggle.com) for the dataset (a competition, see step 3).
- Optional: [Groq](https://groq.com) for audit explanations, [Logfire](https://logfire.pydantic.dev) for tracing. Without `GROQ_API_KEY` the dashboard shows "explanation unavailable". Without `LOGFIRE_TOKEN` nothing is sent, but spans still print to the console.

### 2. Install

```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `DATABASE_URL` and the `MLFLOW_TRACKING_*` values at minimum. Python 3.12 is assumed.

### 3. Dataset

This is a Kaggle *competition* dataset: you must accept the rules on the site, and the plain dataset API returns a 403 even with valid credentials. Download it from the [competition page](https://www.kaggle.com/c/GiveMeSomeCredit/data) and place `cs-training.csv` in `data/raw/`.

### 4. Database

No local `psql` needed. Paste [`src/db/schema.sql`](src/db/schema.sql) into Neon's SQL Editor, or let `run_demo_loop.py` apply it on first run. It is idempotent (`CREATE TABLE IF NOT EXISTS`) and runs as a whole through the driver, not split on semicolons.

Neon's plain `postgresql://` connection string is rewritten to `postgresql+psycopg://` (psycopg v3) by `src/db/connection.py`, so either form works in `.env` and CI secrets.

### 5. DVC remote (DagsHub)

DagsHub's DVC remote is not a bucket you name: it is a fixed placeholder URL (`s3://dvc`) proxied through an `endpointurl` pointing at your repo. Skipping either gives confusing errors.

Your DagsHub repo page has these commands pre-filled with your username, repo and a token (**Remote** button, **Data** tab, **DVC**):

```
dvc init
dvc remote add -d origin s3://dvc
dvc remote default origin
dvc remote modify origin endpointurl https://dagshub.com/<user>/<repo>.s3
dvc remote modify origin --local access_key_id <dagshub-token>
dvc remote modify origin --local secret_access_key <dagshub-token>
```

- **Same token for both fields**, not a username and token pair.
- **Run `dvc remote default origin` anyway.** It is idempotent and cheap insurance after `-d`.
- **Use a token from Settings, then Tokens.** The token shown inline on the Remote page can be a short-lived session token. An account token is long-lived and survives a scheduled job.

The processed pickles come from the demo loop, so run it once (see "Running it") before tracking them:

```
dvc add data/raw/cs-training.csv data/processed/pretrain_batches.pkl data/processed/training_pool.pkl data/processed/holdout.pkl
git add data .dvc .dvcignore
git commit -m "Track data with DVC"
git push
dvc push
```

Verify the upload instead of trusting the exit code:

```
dvc status -c   # should report nothing pending against the remote
```

`dvc add` also writes a `.gitignore` beside the data, which keeps the pickles and CSV out of git, and `git add data` stages it with the `.dvc` pointers. The cron workflow refuses to run until all three processed `.dvc` files are committed.

### 6. GitHub Actions secrets (for the scheduled cron job)

`cron_advance.yml` runs `advance_clock.py` on a fresh empty checkout each time, so every credential must be a repository secret (Settings, Secrets and variables, Actions, New repository secret), not just present in your local `.env`:

| Secret name | Value | Notes |
|---|---|---|
| `DATABASE_URL` | Neon connection string | Either scheme works. |
| `MLFLOW_TRACKING_URI` | `https://dagshub.com/<user>/<repo>.mlflow` | If missing, MLflow does not error: it silently falls back to a new local SQLite store on the runner, and every model lookup fails with a confusing "Registered Model not found". |
| `MLFLOW_TRACKING_USERNAME` | Your DagsHub username | |
| `MLFLOW_TRACKING_PASSWORD` | A DagsHub access token | Settings, Tokens. Use a long-lived token. |
| `DVC_ACCESS_KEY_ID` | Your DagsHub token | Same token as the local DVC remote. |
| `DVC_SECRET_ACCESS_KEY` | Your DagsHub token | Same token, both fields. |

If you edit the workflow:

- `dvc pull` needs `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`, because `dvc-s3` is boto3 underneath and boto3 only reads the AWS names. The secrets keep their `DVC_*` names and the workflow maps them for `dvc pull`.
- A preflight step checks that `.dvc/` exists, the three processed `.dvc` files are committed and both DVC secrets are non-empty, failing with a plain message instead of DVC's generic "not inside of a DVC repository".

Trigger the workflow manually first (Actions, Advance pipeline clock, Run workflow) for fast feedback on misconfiguration.

The schedule is every 3 hours, which finishes the 318 batches in about 40 days. GitHub disables scheduled workflows on public repositories after 60 days without repository activity, so a slower cadence can stall before the stream ends. After that, each run reports `past_end_of_dataset` and exits cleanly.

## Running it

```
python scripts/smoke_test_mlflow.py   # connectivity check, seconds not minutes
python scripts/run_demo_loop.py       # bootstrap, 25 ticks, summary
uvicorn src.serving.app:app --reload  # GET /health, GET /model-info, POST /predict
streamlit run dashboard/app.py        # overview, lineage, drift, audit log
```

A 25-tick run took about 90 seconds against local Postgres and a local MLflow store. Against Neon and DagsHub expect longer, dominated by MLflow round trips on each retrain (not timed). It covers the bootstrap champion, persistent drift from batch 10 and the concept-drift window at batches 15 to 19.

- The demo loop refuses to start on a database that already holds pipeline state. To rerun it, reset first:

  ```sql
  TRUNCATE predictions, champion_history, audit_log RESTART IDENTITY;
  UPDATE pipeline_state SET current_batch = 0, version = 0 WHERE id = 1;
  ```

- Existing processed pickles are reused, not regenerated, so DVC-tracked files stay stable.
- Paths resolve against the repository root, so scripts work from any working directory.
- The serving app rechecks the `@production` alias at most every 30 seconds and loads the exact resolved version, so promotions and rollbacks apply without a redeploy.
- `smoke_test_mlflow.py` prints its tracking URI and fails unless it looks like DagsHub, so it cannot pass against a local fallback store.

## Testing

```
pytest tests -v
ruff check src tests scripts dashboard
```

105 tests across 11 files. Unit tests need nothing external. `test_integration_db.py` and `test_integration_pipeline.py` run only when `TEST_DATABASE_URL` points at Postgres, which CI provides. Locally:

```
docker run -d --name crg-pg -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=testdb -p 5432:5432 postgres:16
export TEST_DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/testdb
pytest tests -v
```

The integration tests drop and recreate the four tables in that database, so never point them at real data.

Coverage:

- **Gate** (`test_gate.py`): rejection paths, bootstrap CI reproducibility and symmetry, and a randomized check that promotion equals the significance result.
- **Drift** (`test_drift_detect.py`): real Evidently at a large reference, low-cardinality and constant columns, staleness immune to p-value noise.
- **Rollback and promotion** (`test_promote_rollback.py`): stale suppression, no-candidate flagging, rollback only when the previous champion is significantly better, alias moved last, reconciliation.
- **Pipeline** (`test_pipeline.py`): pool caps and exclusion, cooldown, no retrain before labels are due, skipped rollback check on a promoting tick.
- **Postgres** (`test_integration_db.py`): migrations, single-winner claims, `SKIP LOCKED` under a held lock, labeled-row filtering, payload filters applied before the limit.
- **End to end** (`test_integration_pipeline.py`): multi-tick runs on Postgres, local MLflow and Evidently, asserting no gate-batch leakage into training, the cooldown, alias reconciliation, and clean rollback of a failed tick.
- **Data, drift injection, serving, cache, clock**: disjoint splits, deterministic drift, endpoints and 503 handling, cache TTL and thread safety, single-writer claims.

## Known limitations

- **McNemar is diagnostic only and often unreliable at this batch size.** About 200 rows and 13 positives per batch rarely yield 15 discordant pairs. `details["mcnemar_reliable"]` says so.
- **Drift is measured on the current batch, labels arrive for an earlier one.** With a 3-batch delay the retrain trigger and gate window can sit in different regimes. This is inherent to delayed labels.
- **Detection power is unmeasured.** Whether 200-row batches reliably flag all three persistently drifted features under K-S at alpha 0.05 is untested on the real dataset. The cooldown bounds the cost of misses and false alarms, not their frequency.
- **A long transaction spans MLflow calls.** A dropped connection rolls the tick back safely and it retries from the start.
- **Rollback rarely has anywhere to revert to in a short run.** With few or no promotions in a 25-tick demo, flagged degradations find no earlier champion and only record that fact.
- **The concept-drift blend target is computed per batch**, not from a fixed reference centroid.
- **Processed data is stored as pickle.** Only load pickles from your own DVC remote.
- **Model quality is not the point.** The challenger is a plain logistic regression on purpose. The governance loop is the deliverable.