# Track A — Final Results and Findings

## 1. Evaluation setup

Track A evaluates the fixed temporal split:

- Fit: timesteps 1–30
- Validation: timesteps 31–34
- Test: timesteps 35–49

The final MLflow retrieval returned 54 test-evaluation runs. Deduplication by model, feature set, graph protocol and seed retained 47 unique runs across 11 configurations. There were zero missing seed values.

Random Forest, XGBoost, GCN and GraphSAGE configurations each use five seeds: 7, 21, 42, 123 and 2024. Logistic Regression has one deterministic run per feature set.

## 2. Aggregate test results

| Model | Feature set | Protocol | Runs | Validation AP | Test AP | Test AP SD | Test ROC-AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| XGBoost | Local + Hand | Tabular | 5 | 0.993551 | **0.795679** | 0.001325 | 0.930470 |
| XGBoost | All | Tabular | 5 | 0.989573 | 0.791330 | 0.001398 | 0.928584 |
| Random Forest | All | Tabular | 5 | 0.991155 | 0.783455 | 0.001215 | 0.918470 |
| Random Forest | Local | Tabular | 5 | 0.991471 | 0.777238 | 0.000478 | 0.891789 |
| XGBoost | Local | Tabular | 5 | 0.990441 | 0.775448 | 0.002339 | 0.918041 |
| GraphSAGE | Local | Directed | 5 | 0.866982 | 0.627072 | 0.022531 | 0.883017 |
| GraphSAGE | Local | Undirected | 5 | 0.910717 | 0.609400 | 0.019798 | 0.884346 |
| GraphSAGE | All | Directed | 5 | 0.847117 | 0.596761 | 0.016305 | 0.890314 |
| Logistic Regression | All | Tabular | 1 | 0.642301 | 0.295219 | N/A | 0.880833 |
| GCN | Local | Directed | 5 | 0.689575 | 0.273804 | 0.036276 | 0.822395 |
| Logistic Regression | Local | Tabular | 1 | 0.573005 | 0.217943 | N/A | 0.847810 |

Standard deviations are across random seeds (sample standard deviation). Logistic Regression has one run, so its standard deviation is not defined.

## 3. Main findings

### 3.1 Explicit message passing did not outperform the tabular baselines

The best GNN configuration was GraphSAGE with Local features and directed edges (test AP 0.627072). The strongest tabular configuration, XGBoost with Local + Hand features, achieved test AP 0.795679.

Matched feature-set comparisons also favoured XGBoost:

- Local: XGBoost 0.775448 vs GraphSAGE 0.627072.
- All: XGBoost 0.791330 vs GraphSAGE 0.596761.

For this dataset and evaluation protocol, Track A provides no evidence that explicit GNN message passing improves illicit-transaction ranking over the evaluated tabular baselines.

### 3.2 Handcrafted graph features improved XGBoost

XGBoost test AP increased from 0.775448 with Local features to 0.795679 with Local + Hand features, an absolute increase of approximately 0.020231 AP.

This suggests that the selected graph-derived neighbourhood statistics can add useful information to a tabular model without requiring learned message passing.

### 3.3 Supplied neighbour aggregates were not uniformly beneficial

The All feature set combines 93 Local features and 72 supplied neighbour aggregate features. All improved XGBoost and Random Forest relative to their Local-only versions, but GraphSAGE All performed worse than GraphSAGE Local.

The supplied neighbour-feature provenance is insufficiently documented. Their freedom from label leakage therefore cannot be independently verified; this limitation should accompany interpretation of results involving All.

### 3.4 GraphSAGE outperformed GCN

GraphSAGE Local directed achieved test AP 0.627072, compared with 0.273804 for GCN Local directed. GraphSAGE was the stronger of the two evaluated GNN architectures under this setup.

### 3.5 Symmetrisation did not improve held-out test AP

GraphSAGE Local undirected achieved validation AP 0.910717, higher than the directed configuration's 0.866982. However, its test AP was lower: 0.609400 versus 0.627072.

The validation ranking did not translate into a test-period improvement.

### 3.6 Temporal generalisation remains challenging

The strongest XGBoost configuration's mean validation AP was 0.993551, while its mean test AP was 0.795679. GNN configurations also exhibited validation-to-test declines.

This indicates a substantial difference between validation and later test-period performance across model families. Track A alone does not establish the cause of this temporal degradation.

## 4. Interpretation boundaries and limitations

- Findings apply to the Elliptic dataset and the specified Track A temporal split; they should not be generalised to all graph-learning or financial-crime-detection settings.
- Unknown labels must remain unknown and must not be interpreted as licit.
- The supplied 72 neighbour aggregate features have insufficiently documented provenance; leakage-freedom cannot be independently established.
- Results are ranking/evaluation results on the dataset, not evidence of production AML effectiveness or identification of real-world criminal actors.
- Logistic Regression results are single-run deterministic baselines; no across-seed variability estimate is available.
- The Track A comparison does not by itself establish statistical significance between models. Reported standard deviations describe variation across the five chosen seeds, not confidence intervals over independent datasets.

## 5. Track A conclusion

On the fixed Track A temporal split, tree-based tabular models substantially outperformed the evaluated GNNs. XGBoost with Local + Hand features was the strongest configuration. Handcrafted graph-derived features improved XGBoost over Local-only features, while explicit GNN message passing did not provide a test-performance advantage.

These results motivate Track B, which tests whether the observed ranking and temporal degradation persist across multiple forward-in-time evaluation windows. Track B should use the already selected hyperparameters without tuning against its test windows.

## 6. MLflow provenance

- Experiments retrieved: `elliptic-track-a-gnn` (ID 8), `elliptic-track-a-xgboost` (ID 4), `elliptic-track-a-tabular-baselines` (ID 3).
- Test runs retrieved: 54.
- Unique runs retained after deduplication: 47.
- Unique configurations: 11.
- Missing seed values after extraction: 0.
- Deduplication key: `(model, feature_set, graph_protocol, seed)`, retaining the latest run for each key.
- GNN seed parameter: `random_seed`.
- Tabular seed parameter: `random_state`.
