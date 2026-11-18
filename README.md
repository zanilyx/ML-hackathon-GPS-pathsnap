PathSnap: GPS map-matching and route reconstruction
Team: Members: <name 1>, <name 2>, <name 3>, <name 4> Course: UE24CS352A Machine Learning, PES University (PathSnap Hackathon, 18-11-2026)

PathSnap takes noisy GPS points from a vehicle and works out which roads it actually drove on. A classifier scores candidate road segments for every GPS point, and an HMM with Viterbi decoding then joins those picks into a connected route. The result is served through a FastAPI backend and shown in a separate UI.

Status
 Dataset exploration (EDA, Folium map)
 Candidate generation and feature engineering
 Six models trained and compared (trip-grouped CV + student-test)
 Bayesian hyperparameter optimisation (XGBoost, Optuna)
 HMM / Viterbi route reconstruction
 FastAPI backend
 Evaluator UI
 Technical report (PDF)
Setup
Python 3.10+ is needed. Install the dependencies:

pip install -r requirements.txt
Put the dataset in a folder called Student folder/ next to the scripts. It should contain train/ and student_test/, each with one sub-folder per trip. The dataset is not in this repo. We never read the evaluator-testing partition during development.

How to run
Run these from the repo root, in this order:

py eda.py                  # dataset inventory, anomalies, plots, Folium map  -> outputs/eda/
py make_features.py        # candidates + features for train and student_test  -> outputs/features.pkl
py train_compare.py        # trains and compares the models                    -> outputs/
py diagnose_misses.py      # per-trip candidate miss analysis (optional)
Each script writes live progress to outputs/dashboard.html. Open it in a browser while a script runs. train_compare.py uses the GPU for XGBoost if CUDA is available and falls back to the CPU otherwise.

What the code does
Projection. Longitude/latitude are converted to metres with a local equirectangular projection centred on each trip's road network, so all distances are in metres and not degrees.

Candidates. For each GPS point we take the 10 nearest road segments (K = 10) within 150 m, always keeping the nearest one. Distances are exact point-to-segment distances. For speed and memory, points are handled in chunks of 256 and each chunk is only compared against the arcs near it.

Label. A candidate is positive if its arc appears in the trip's .route file. The route is a sequence of arcs with no per-point alignment, so this is a proxy label. It can be wrong when a route revisits a road or when a parallel road is very close.

Features (16). Distance to the arc, position along the arc, arc length, distance rank, distance minus the nearest candidate's distance, distance ratio, number of candidates, heading agreement with the arc direction (signed and absolute), whether the heading is valid, speed, time since the previous point, distance of the previous and next GPS point to the same arc, and node degree at both arc ends.

Cross-validation. GroupKFold with 5 folds, grouped by trip, so no trip is ever split between training and validation. Scalers are inside sklearn pipelines and fitted on the training fold only. Everything uses seed 42.

Metrics.

Macro F1 on candidate labels (threshold 0.5) and log loss.
Observation-level Top-1: the highest-scored candidate for each GPS point is correct if its arc is in the reference route.
Candidate miss rate: the share of GPS points where no candidate is in the reference route.
Route continuity: among consecutive distinct predicted arcs, the share where the previous arc's end node is the next arc's start node.
Training time and inference time.
Model comparison (so far)
CV is 5-fold grouped CV on the 80 training trips. Test is the 10 student-testing trips.

Model	Family	CV macro F1	Test macro F1	Test Top-1	Test log loss	Test continuity	Train time
Logistic Regression	Linear	0.757	0.760	0.862	0.526	0.550	1.5 s
Gaussian Naive Bayes	Probabilistic	0.723	0.720	0.869	0.809	0.601	0.4 s
XGBoost (GPU)	Boosting	0.807	0.787	0.885	0.523	0.557	3.4 s
HistGradientBoosting	Boosting	0.808	0.786	0.887	0.512	0.593	24.6 s
MLP (64, 32)	Neural network	0.817	0.792	0.881	0.528	0.446	184 s
Random Forest	Ensemble	0.809	0.790	0.888	0.521	0.569	45 s
The full numbers (fold-by-fold results, inference times, miss rates) are in outputs/model_comparison.csv and outputs/cv_fold_results.csv.

Reading the table. The linear and Naive Bayes models are clearly behind. XGBoost, HistGradientBoosting, Random Forest and the MLP are within about 0.006 of each other on test F1, which is inside the noise with only 10 test trips. The MLP had the lowest route continuity in all five CV folds, so we dropped it. We picked XGBoost to take forward into tuning because it trains in a few seconds on the GPU, its probabilities are close to the best log loss, and its fold-to-fold spread is small. This choice is provisional until the tuning and HMM results are in.

Known data issues
Candidate misses. On the student-test trips only 0.07% of GPS points have no correct candidate, but a few training trips are much worse (trips 33 and 54 miss over half of their points). diagnose_misses.py shows two causes: on trips 33 and 54 the GPS sits a steady 120 to 140 m from the reference route (a systematic offset), and on trips such as 72 and 61 the correct arc exists but falls outside the 10 nearest candidates. This affects every model equally, so it does not change the ranking.
Time gaps. One training trip has a gap of about 153,000 s between points, and the share of 1 s steps drops as low as 53% in some trips.
Fragmented road networks. Several trips have dozens of disconnected components (up to 421), so the HMM will meet disconnected candidate pairs often.
Reference routes are treated as reference information, not guaranteed ground truth.
Repository layout
common.py             projection, candidate generation, metrics
eda.py                dataset exploration and the Folium map
make_features.py      builds the candidate/feature tables
train_compare.py      trains and compares the models
diagnose_misses.py    candidate-miss analysis per trip
telemetry.py          progress dashboard used by the scripts
requirements.txt
outputs/              results: CSVs, plots, config, Folium map, small saved models
Large files (features.pkl, the bigger model files) are left out of the repo and can be regenerated by the scripts above.

AI tool use
We used Claude (Anthropic) as an assistant while building this project, for help with code and planning. Chat link for the TA:

Results, metrics and plots in this repo come from our own runs of the code. We rewrote the explanations in our own words and every team member should be able to explain the code and results.

Notes
No credentials or API keys are stored in this repository.
Evaluator ground truth is never part of the prediction pipeline or the UI.