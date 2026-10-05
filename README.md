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

A model does not stay good just because it was good at launch. This project simulates the world drifting after deployment, and the pipeline that watches, challenges and replaces the model automatically.

On every clock tick:

1. Reset `@production` to the champion recorded in Postgres, then score the batch.
2. Check drift against the base training pool (Evidently, K-S test per feature).
3. Release labels for the earlier batch whose delay has elapsed.
4. If drift crosses the threshold and the cooldown has elapsed, train a challenger (capped base sample plus recent labeled rows, excluding the gate batch) and compare it to the champion on that batch.
5. Promote only if the bootstrap CI on the metric delta clears zero.
6. Screen the champion for degradation against the stored reference, and roll back only if the previous champion is significantly better on the same live batch.

Every decision lands in an audit log. The loop runs on a schedule via GitHub Actions.

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

Gate, drift and retrain values live in `config/gate_config.yaml`. The label delay lives in `config/drift_params.yaml`.

| Parameter | Value | Why |
|---|---|---|
| Primary metric | AUC-PR | Accuracy is near meaningless at a ~6.7% positive rate. |
| Decision threshold | 0.5 | Used for label predictions and threshold-based metrics. |
| Significance | Bootstrap 95% CI (2.5th percentile) on the matched-batch delta above 0, 2,000 resamples | The deciding test. |
| Tolerance band | 0.01 | First rejection reason: worse than champion by more than this. |
| Dominance | challenger > champion | Second rejection reason: a tie is rejected. |
| McNemar | diagnostic only, reliable at 15+ discordant pairs | Logged, never decides an outcome. |
| Drift test | K-S p-value < 0.05 per feature | Forced, with every feature declared numerical. Evidently's default changes test with reference size. |
| Drift share threshold | 0.3 | 3+ of 10 features flagging by chance is rare under K-S at alpha 0.05 (about 1% per undrifted batch, assuming independence). |
| Retrain cooldown | 3 batches | Drift is measured against the base pool, so persistent drift never clears. Without a cooldown, every tick would retrain. |
| Retrain data | 3,000 base rows + up to 6,000 most recent labeled rows | The gate batch is always excluded. |
| Rollback screen | drop of 0.03 AUC-PR versus the stored reference | Cheap trigger only. The reference is one upward-biased gated batch, so it never decides a rollback alone. |
| Rollback decision | previous champion beats current on the same live batch, bootstrap CI lower bound above 0 | Same rigor as promotion, identical rows for both models. |
| Staleness | drift share moved by more than 0.3, or more than 30% of features disagree on drifted or not | Compares which features drifted. Raw p-values are uniform noise when nothing has drifted. |
| Delayed labels | 3 batches | Nothing is scored against labels that would not yet exist. |

Promotion requires the tolerance, dominance and significance checks to all pass. Significance almost always implies the other two, which are kept as ordered rejection reasons for specific audit messages. A randomized test checks that promotion tracks the significance result.

## Design invariants

- **No leakage.** The stream is carved out before the base pool, retrains exclude the gate batch, and imputation medians come from the base pool only.
- **Significance over raw gaps.** About 13 positives per batch make raw AUC-PR differences noise, so the bootstrap CI must clear zero.
- **Paired rollback.** The stored reference only screens. The decision compares both models on the same live batch.
- **Pinned drift test.** K-S is forced because Evidently switches to Wasserstein above 1,000 reference rows and misses heavy tails like `DebtRatio`. Staleness compares drifted feature sets, not p-values.
- **Postgres decides the champion.** Alias moves run last and are reconciled every tick. `SKIP LOCKED` claims never block.
- **Challengers can adapt.** The base pool is sampled down and labeled rows are capped.
- **Guarded triggers.** Retraining needs labels and an elapsed cooldown. A promoting tick skips the rollback check.
- **No target in features.** Predictions store inputs only. `true_label` is filled on release.

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
- `champion_history`: promotion lineage with window metrics, frozen-holdout metrics, drift fingerprint, staleness flag and rollback markers.
- `audit_log`: every governance decision by event type: `clock_advance`, `drift_check`, `label_release`, `gate_evaluation`, `promotion`, `rollback_check`, `rollback`, `alias_reconciled`.

Models and registry: MLflow on DagsHub, alias-based (`@production`, never the deprecated stage API). Data: DVC against a DagsHub-hosted S3-compatible remote.

The dashboard trend chart and lineage deltas use the frozen-holdout metric, the only value measured on the same rows for every champion. The per-champion window metric is shown separately as the reference window.

## LLM explanations

The audit log tab has an on-demand "Explain in plain language" button that sends one event's type and payload to Groq and shows 2 to 3 sentences. It is read-only and never influences a decision. The output is grounded only in the payload and can still be wrong, so treat it as a reading aid, not a source of truth. Without `GROQ_API_KEY` the dashboard shows "explanation unavailable".

## Project structure

```
credit-risk-autopilot/
├── assets/                     # dashboard icon and README screenshots
│
├── config/
│   ├── drift_params.yaml       # recession scenario, label delay
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
├── .dvc/                       # DVC remote config
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
- [Kaggle](https://www.kaggle.com) for the dataset.
- Optional: [Groq](https://groq.com) for audit explanations, [Logfire](https://logfire.pydantic.dev) for tracing. Logfire spans cover the serving `/predict` endpoint only. Without `LOGFIRE_TOKEN` nothing is sent.

### 2. Install

```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `DATABASE_URL` and the `MLFLOW_TRACKING_*` values at minimum. Python 3.12 is assumed. DVC credentials are not read from `.env`; locally DVC uses the `--local` config from step 5, and CI uses repository secrets.

### 3. Dataset

This is a Kaggle competition dataset: you must accept the rules on the site, and the plain dataset API returns a 403 even with valid credentials. Download `cs-training.csv` from the [competition page](https://www.kaggle.com/c/GiveMeSomeCredit/data) and place it in `data/raw/`.

### 4. Database

No local `psql` needed. Paste [`src/db/schema.sql`](src/db/schema.sql) into Neon's SQL Editor, or let `run_demo_loop.py` apply it on first run. It is idempotent.

A plain `postgresql://` connection string is rewritten to `postgresql+psycopg://` by `src/db/connection.py`, so either form works.

### 5. DVC remote (DagsHub)

DagsHub's DVC remote is a fixed placeholder URL (`s3://dvc`) proxied through an `endpointurl` pointing at your repo. Your repo page has these commands pre-filled (**Remote** button, **Data** tab, **DVC**):

```
dvc init
dvc remote add -d origin s3://dvc
dvc remote default origin
dvc remote modify origin endpointurl https://dagshub.com/<user>/<repo>.s3
dvc remote modify origin --local access_key_id <dagshub-token>
dvc remote modify origin --local secret_access_key <dagshub-token>
```

Use the same token for both fields, and take it from Settings, then Tokens. The token shown inline on the Remote page can be short-lived.

The processed pickles come from the demo loop, so run it once (see "Running it") before tracking them:

```
dvc add data/raw/cs-training.csv data/processed/pretrain_batches.pkl data/processed/training_pool.pkl data/processed/holdout.pkl
git add data .dvc .dvcignore
git commit -m "Track data with DVC"
git push
dvc push
dvc status -c
```

`dvc status -c` should report nothing pending. The cron workflow refuses to run until all three processed `.dvc` files are committed.

## Running it

```
python scripts/smoke_test_mlflow.py
python scripts/run_demo_loop.py
uvicorn src.serving.app:app --reload
streamlit run dashboard/app.py
```

- `smoke_test_mlflow.py` fails unless the tracking URI looks like DagsHub, so it cannot pass against a local fallback store.
- `run_demo_loop.py` bootstraps a champion, runs 25 ticks and prints a summary. It covers persistent drift from batch 10 and the concept-drift window at batches 15 to 19. A run took about 90 seconds against local Postgres and a local MLflow store. Against Neon and DagsHub expect longer (not timed).
- The demo loop refuses to start on a database that already holds pipeline state. To rerun it, reset first:

  ```sql
  TRUNCATE predictions, champion_history, audit_log RESTART IDENTITY;
  UPDATE pipeline_state SET current_batch = 0, version = 0 WHERE id = 1;
  ```

- Existing processed pickles are reused, not regenerated, so DVC-tracked files stay stable.
- The serving app exposes `GET /health`, `GET /model-info` and `POST /predict`. It rechecks the `@production` alias at most every 30 seconds and loads the exact resolved version, so promotions and rollbacks apply without a redeploy.

## Scheduled deployment

`cron_advance.yml` runs `advance_clock.py` every 3 hours on a fresh checkout, which finishes the 318 batches in about 40 days. After that, each run reports `past_end_of_dataset` and exits cleanly. GitHub disables scheduled workflows on public repositories after 60 days without repository activity, so a slower cadence can stall before the stream ends.

Every credential must be a repository secret (Settings, Secrets and variables, Actions):

| Secret | Value | Notes |
|---|---|---|
| `DATABASE_URL` | Neon connection string | Either scheme works. |
| `MLFLOW_TRACKING_URI` | `https://dagshub.com/<user>/<repo>.mlflow` | If missing, MLflow silently falls back to a new local SQLite store and every model lookup fails with "Registered Model not found". |
| `MLFLOW_TRACKING_USERNAME` | DagsHub username | |
| `MLFLOW_TRACKING_PASSWORD` | DagsHub access token | Use a long-lived token. |
| `DVC_ACCESS_KEY_ID` | DagsHub token | The workflow maps it to `AWS_ACCESS_KEY_ID` for `dvc pull`. |
| `DVC_SECRET_ACCESS_KEY` | DagsHub token | Mapped to `AWS_SECRET_ACCESS_KEY`. |

A preflight step fails with a plain message if `.dvc/`, the three processed `.dvc` files or the DVC secrets are missing. Trigger the workflow manually first (Actions, Advance pipeline clock, Run workflow) to catch misconfiguration.

## Failure behavior

A tick that raises rolls back its claim and all database rows, so the next run retries the same batch. MLflow is not transactional: runs and model versions registered by a failed retrain remain as orphans and are never promoted, because the alias is reconciled to the database champion on the next tick. A racing run that loses the claim exits without work.

## Testing

```
pytest tests -v
ruff check src tests scripts dashboard
```

105 tests across 11 files. 89 need nothing external. `test_integration_db.py` and `test_integration_pipeline.py` run only when `TEST_DATABASE_URL` points at Postgres, which CI provides. Locally:

```
docker run -d --name crg-pg -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=testdb -p 5432:5432 postgres:16
export TEST_DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/testdb
pytest tests -v
```

The integration tests drop and recreate the four tables in that database, so never point them at real data. The end-to-end tests also assert no gate-batch leakage into training, the cooldown, alias reconciliation and clean rollback of a failed tick.

## Known limitations

- **Low power.** About 200 rows and 13 positives per batch widen the CI, so real improvements are often rejected, rollback rarely has an earlier champion, and McNemar rarely reaches 15 discordant pairs.
- **Label lag.** Drift is measured on the current batch but labels belong to an earlier one, so trigger and gate can sit in different regimes.
- **Detection power unmeasured.** K-S on 200 rows may miss drifted features. The cooldown bounds the cost, not the frequency.
- **Long transactions.** A tick's transaction spans MLflow calls. A dropped connection rolls it back and it retries.
- **Scope.** Per-batch concept-drift centroid, pickle storage (trusted remote only), deliberately plain logistic regression.
