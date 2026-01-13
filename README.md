# ammoniacal-nitrogen-prediction-tiete
Machine learning models (Random Forest, XGBoost, and MLP) for ammoniacal nitrogen prediction to support water-quality monitoring in the Tietê River (CETESB station TIET04200).
# Ammoniacal Nitrogen Prediction (Tietê River) — RF, XGBoost, MLP

Machine learning models to **predict ammoniacal nitrogen (NH₃/NH₄⁺)** at the **TIET04200** monitoring station (Tietê River, UGRHI 6 – Upper Tietê Basin, Brazil) using **CETESB historical data (2019–2025)**, under a **leakage-free temporal pipeline** and **time-aware validation**.

> Beyond prediction/nowcasting support for water-quality monitoring, the framework can also work as an **additional QA/QC layer**, flagging potentially inconsistent records using large residuals and their temporal behavior.

---

## 📌 Project goals

- Predict ammoniacal nitrogen (mg/L) in an urban, impacted river reach (TIET04200).
- Compare **Random Forest (RF)**, **XGBoost**, and **Multilayer Perceptron (MLP)** under the same temporal protocol.
- Evaluate performance using **R², RMSE, MAE, MAPE, and CCC**.
- Simulate operational behavior using **dynamic backtesting** on the most recent samples.

---

## 🌎 Study area and data

- **Station:** TIET04200 (Ponte dos Remédios, Marginal Tietê), São Paulo Metropolitan Region (RMSP).
- **Basin:** UGRHI 6 – Alto Tietê (Upper Tietê Basin).
- **Data source:** CETESB (InfoÁguas portal) — public historical monitoring data (Excel).
- **Training window (modeling):** 2019–2025  
- **Seasonality analysis (exploratory):** 2010–2025 (used only for seasonal plots/tests in the paper)

---

## 🧠 Models evaluated

- **Random Forest (RF)** — ensemble of decision trees (Breiman, 2001)
- **XGBoost** — gradient boosted trees (Chen & Guestrin, 2016)
- **MLP** — feedforward neural network (Rumelhart et al., 1986)

**Hyperparameters (paper):**
- **MLP:** `hidden_layer_sizes=(100, 50)`, `max_iter=2000`, `random_state=42`
- **XGBoost:** `n_estimators=500`, `learning_rate=0.05`, `max_depth=5`,  
  `subsample=0.9`, `colsample_bytree=0.9`, `reg_lambda=1.0`, `objective=reg:squarederror`, `random_state=42`
- **RF:** `n_estimators=400`, `random_state=42`, `n_jobs=-1`

---

## 🧼 Leakage-free temporal pipeline

All data-driven steps are **fitted only on the training split within each temporal fold** (no future leakage):

1. **Cleaning** (invalid dates, duplicates, typing/unit inconsistencies)
2. **Missing values:** forward-fill; remaining NAs → training median
3. **Outlier attenuation:** RobustClipper (±3 std per column)
4. **Seasonality features:** month + season dummies (0/1)
5. **Feature screening:** TOP_K = 5 by |Spearman ρ| with the target (computed on training only)
6. **Standardization:** fit on training, apply to test (kept for pipeline consistency)

---

## ⏱️ Validation strategy

- **OOF temporal CV:** `TimeSeriesSplit (n=5)` (forward chaining)
- **Dynamic backtesting (operational simulation):**
  - **Recent window:** last 10 samples (**2023-02-08 → 2025-05-06**)
  - **Extended window (stress test):** 16 samples (**2021-09-22 → 2025-05-06**)

---

## 📊 Main results (from the paper)

### Temporal CV (OOF, 2019–2025)

| Model | R² | RMSE (mg/L) | MAE (mg/L) | MAPE (%) | CCC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.8361 | 2.8320 | 2.1320 | 13.7710 | 0.9044 |
| XGBoost | 0.8309 | 2.8761 | 2.2848 | 15.4218 | 0.9069 |
| MLP | 0.8113 | 3.0384 | 2.4218 | 21.5155 | 0.9103 |

### Dynamic backtesting — last 10 samples (2023-02-08 → 2025-05-06)

| Model | R² | RMSE (mg/L) | MAE (mg/L) | MAPE (%) | CCC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.9130 | 2.1344 | 1.8360 | 14.8849 | 0.9524 |
| XGBoost | 0.9024 | 2.2603 | 1.9102 | 14.0917 | 0.9496 |
| MLP | 0.7337 | 3.7341 | 3.1696 | 38.0493 | 0.8368 |

### Dynamic backtesting — extended window (2021-09-22 → 2025-05-06)

| Model | R² | RMSE (mg/L) | MAE (mg/L) | MAPE (%) | CCC |
|---|---:|---:|---:|---:|---:|
| MLP | 0.7973 | 3.2200 | 2.8131 | 26.5992 | 0.8853 |
| Random Forest | 0.7438 | 3.6199 | 2.8211 | 16.1055 | 0.8370 |
| XGBoost | 0.7268 | 3.7380 | 3.0141 | 17.1075 | 0.8442 |

**Key takeaway:** RF and XGBoost are strongest in the **recent operational window**, while the extended window highlights **temporal regime sensitivity** (concept drift).

---

## 🧪 Metrics

- **MAE** (Mean Absolute Error)
- **MSE** and **RMSE**
- **R²** (Coefficient of Determination)
- **MAPE** (Mean Absolute Percentage Error)
- **CCC** (Concordance Correlation Coefficient)

---

## 🗂️ Suggested repository structure

> Adjust names to match your current files.

```text
.
├─ data/
│  ├─ raw/            # CETESB exports (do not commit if large)
│  └─ processed/      # cleaned tables used by training
├─ notebooks/         # EDA, correlation plots, seasonal analysis
├─ src/
│  ├─ preprocessing.py
│  ├─ features.py
│  ├─ train.py
│  ├─ evaluate.py
│  └─ backtesting.py
├─ outputs/
│  ├─ figures/
│  └─ tables/
├─ requirements.txt
└─ README.md

##Recommended packages

numpy, pandas

scikit-learn

xgboost

matplotlib (and seaborn if you use it in notebooks)

(optional) tensorflow/keras if you run NN experiments


##▶️ How to run (typical workflow)

Put CETESB/InfoÁguas export in data/raw/

Run preprocessing:

python -m src.preprocessing


Train + temporal CV:

python -m src.train --model rf
python -m src.train --model xgb
python -m src.train --model mlp


Dynamic backtesting:

python -m src.backtesting --window recent
python -m src.backtesting --window extended


Generate figures/tables:

python -m src.evaluate


## ✅ QA/QC use-case (optional but recommended)

This framework can be used as an additional data-quality checkpoint:

flag samples with large residuals (observed − predicted),

prioritize retesting / audits (unit/typing errors, contamination, preservation issues, analytical interferences),

track residual drift over time as a sentinel of process bias.


##📝 Citation

If you use this repository, please cite the related manuscript:

Pacheco, G. N.; Silva, J. C.; Andrade, R. C.; Silva Filho, P. A.
Machine learning for ammoniacal nitrogen prediction to support water-quality monitoring in the Tietê River: Random Forest, XGBoost, and Multilayer Perceptron.


##👤 Authors

Gustavo Nunes Pacheco (main author)

Júlio César da Silva (advisor)

Rosane Cristina de Andrade (text review + figures)

Pedro Alves da Silva Filho (text review)

##📄 License

Choose a license (e.g., MIT, Apache-2.0) and add a LICENSE file.
If data redistribution is restricted, keep raw data out of the repo and document how to obtain it.



##📬 Contact

Gustavo Nunes Pacheco
Email: gustavoo.np@hotmail.com


