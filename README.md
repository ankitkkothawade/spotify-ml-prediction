# 🎵 Spotify ML Prediction Project  
**Author:** Ankit Kothawade  
**Domain:** Music Data Science | Regression & Classification  
**Tech Stack:** Python, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, XGBoost, LightGBM  

---

## 📌 Overview  
This project focuses on **predicting song attributes and genres using machine learning** on a Spotify dataset.  
It includes two complementary ML pipelines:

1. 🎯 **Regression:** Predicting a song’s *popularity score* (`pop`)  
2. 🧩 **Classification:** Predicting the *genre* (`top_genre`)  

Both pipelines are built for **robustness, interpretability, and recruiter visibility**, including preprocessing, model tuning, evaluation metrics, and visualizations.

---

## ⚙️ Features  
✅ Clean, modular Python scripts (`src/`) following ML engineering best practices  
✅ Automated preprocessing — handles missing values, scaling, encoding  
✅ Randomized hyperparameter search across multiple model families  
✅ Cross-validation and holdout metrics with visual leaderboard charts  
✅ Compatible with older scikit-learn versions  
✅ Exports artifacts: metrics, plots, and reports (in `/results`)  

---

## 📂 Repository Structure  

spotify-ml-prediction/
├── data/ # Training and testing datasets
│ ├── RegressionTrain.csv
│ ├── RegressionTest.csv
│ ├── ClassificationTrain.csv
│ ├── ClassificationTest.csv
│
├── src/ # ML pipeline source scripts
│ ├── regression_pipeline_pro.py
│ ├── classification_pipeline_pro.py
│
├── results/ # Auto-generated metrics & plots
│ ├── regression_metrics.csv
│ ├── regression_r2_comparison.png
│ ├── regression_pred_vs_true.png
│ ├── classification_metrics.csv
│ ├── classification_confusion.png
│ ├── classification_accuracy_comparison.png
│
└── README.md # Project documentation

---

## 🎯 Regression Task — Predicting Song Popularity

**Target Variable:** `pop`

**Best Model:** `GradientBoostingRegressor`  
**Cross-validated R² (best):** 0.333  
**Holdout R²:** **0.491**  
**Holdout RMSE:** 10.67  
**Holdout MAE:** 8.43  

### Artifacts
| Type | File |
|------|------|
| CV Leaderboard | [`results/cv_leaderboard.csv`](results/cv_leaderboard.csv) |
| Best Model Name | [`results/best_model_name.json`](results/best_model_name.json) |
| Best Hyperparameters | [`results/best_params.json`](results/best_params.json) |
| Holdout Metrics | [`results/regression_metrics.csv`](results/regression_metrics.csv) |
| CV R² Plot | ![CV R²](results/regression_r2_comparison.png) |
| Predicted vs True Plot | ![Pred vs True](results/regression_pred_vs_true.png) |

---

## 🧩 Classification Task — Predicting Song Genre

**Target Variable:** `top_genre`  
**Best Model:** `RandomForestClassifier`  
**Cross-validated Accuracy (best):** 0.386  
**Holdout Accuracy:** *(see `results/classification_metrics.csv`)*  
**Holdout F1 (weighted):** *(see `results/classification_metrics.csv`)*  

> ⚖️ Rare genres with fewer than 3 samples were consolidated into an “other” class to ensure balanced stratified evaluation.

### Artifacts
| Type | File |
|------|------|
| CV Leaderboard | [`results/classification_cv_leaderboard.csv`](results/classification_cv_leaderboard.csv) |
| Best Model Name | [`results/classification_best_model_name.json`](results/classification_best_model_name.json) |
| Best Hyperparameters | [`results/classification_best_params.json`](results/classification_best_params.json) |
| Holdout Metrics | [`results/classification_metrics.csv`](results/classification_metrics.csv) |
| CV Accuracy Plot | ![CV Accuracy](results/classification_accuracy_comparison.png) |
| Confusion Matrix | ![Confusion Matrix](results/classification_confusion.png) |

---

## 🧠 Methodology Summary

| Stage | Description |
|--------|--------------|
| **Data Cleaning** | Dropped NaN targets, imputed missing features (median for numeric, most-frequent for categorical) |
| **Feature Engineering** | Normalized columns, label encoded categorical targets |
| **Preprocessing** | StandardScaler + OneHotEncoder (via `ColumnTransformer`) |
| **Model Families** | Linear, SVR, RandomForest, GradientBoosting, XGBoost, LightGBM |
| **Hyperparameter Optimization** | RandomizedSearchCV with 5-fold CV |
| **Evaluation Metrics** | R² / RMSE / MAE (Regression), Accuracy / F1 / Confusion Matrix (Classification) |
| **Result Logging** | Metrics saved to `.csv` and `.json` files, charts auto-generated |

---

## 📊 Example Outputs

**Regression: Predicted vs True Popularity**
![Pred vs True](results/regression_pred_vs_true.png)

**Classification: Confusion Matrix**
![Confusion Matrix](results/classification_confusion.png)

---

## 🧩 Tech Stack

| Category | Tools |
|-----------|-------|
| **Languages** | Python (3.9) |
| **Libraries** | Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn |
| **Optional** | XGBoost, LightGBM |
| **Environment** | MacOS, Command-line execution |
| **Versioning** | Git, GitHub |

---

## 💡 Key Highlights

- End-to-end ML pipelines following **best MLOps structure**
- Automatically handles **imbalanced classes and missing values**
- Uses **cross-validation** for reliable model selection
- Exports all experiment artifacts for reproducibility
- Designed to demonstrate **Data Science engineering readiness**

---

### ⚙️ How to Run the Project

You can execute and reproduce all results directly from your local environment in just a few steps:

#### **1️⃣ Install Dependencies**

Ensure Python ≥ 3.9 is installed, then install all required libraries:

```bash
pip install -r requirements.txt
```

> 💡 *(You can install globally or inside a virtual environment — both are supported.)*

---

#### **2️⃣ Prepare the Data**

Keep your CSV datasets inside the `data/` folder:

```
data/
 ├── RegressionTrain.csv
 ├── RegressionTest.csv
 ├── ClassificationTrain.csv
 └── ClassificationTest.csv
```

Each dataset should include Spotify song features such as
`dur, nrgy, dnce, dB, live, val, acous, spch, pop, top_genre, year, artist, title`.

---

#### **3️⃣ Run the Regression Pipeline**

Predicts song **popularity (`pop`)** using acoustic and musical features.

```bash
python3 src/regression_pipeline_pro.py \
  --train data/RegressionTrain.csv \
  --test  data/RegressionTest.csv \
  --target pop \
  --results_dir results
```

---

#### **4️⃣ Run the Classification Pipeline**

Predicts the **music genre (`top_genre`)** based on song characteristics.

```bash
python3 src/classification_pipeline_pro.py \
  --train data/ClassificationTrain.csv \
  --test  data/ClassificationTest.csv \
  --target top_genre \
  --results_dir results
```

---

#### **5️⃣ View Results**

All evaluation outputs, metrics, and plots will automatically be saved in the `results/` directory:

```
results/
 ├── regression_metrics.csv
 ├── regression_r2_comparison.png
 ├── regression_pred_vs_true.png
 ├── classification_metrics.csv
 ├── classification_accuracy_comparison.png
 ├── classification_confusion.png
 └── classification_report.txt
```

You can open these files directly or visualize them inside **Jupyter**, **VS Code**, or any Python environment.

---


### 📊 Expected Output

Once the scripts finish execution, the console will display model performance summaries like this:

#### **Regression Output Example**

```
✅ Done.
Best model: GradientBoosting
           model       R2      RMSE      MAE
GradientBoosting 0.333     10.67     8.43
RandomForest     0.324     10.83     8.35
LinearRegression 0.315     10.80     8.94
SVR_rbf          0.291     10.95     8.43

Holdout R²: 0.491
Holdout RMSE: 10.67
Holdout MAE: 8.43
```

This means the **Gradient Boosting model** achieved the best generalization on unseen test data, with R² ≈ **0.49**, indicating a moderate fit and predictive stability.

---

#### **Classification Output Example**

```
✅ Done.
Best model: RandomForest
       model  CV_Accuracy
RandomForest     0.3857
SVM_rbf          0.3571

Holdout Accuracy: 0.39
Holdout F1 (weighted): 0.37
```

Here, the **Random Forest classifier** performed best, achieving ≈ **38.6% cross-validated accuracy** after balancing and consolidating rare genres.

---

### 🗂️ Result Artifacts

All generated reports, charts, and metrics are stored automatically in the `results/` folder.

| File                                     | Description                                     |
| ---------------------------------------- | ----------------------------------------------- |
| `regression_metrics.csv`                 | Final model metrics (R², RMSE, MAE)             |
| `regression_r2_comparison.png`           | Model comparison chart                          |
| `regression_pred_vs_true.png`            | Scatter plot of predicted vs. actual popularity |
| `classification_metrics.csv`             | Final accuracy and F1 scores                    |
| `classification_accuracy_comparison.png` | CV accuracy leaderboard                         |
| `classification_confusion.png`           | Confusion matrix for genre predictions          |
| `classification_report.txt`              | Detailed precision, recall, and F1 per genre    |

---

### 🧾 Conclusion

This project demonstrates the application of **machine learning and data-driven insights** in the music domain, leveraging both regression and classification tasks to analyze and predict song characteristics.

* **Regression Pipeline** showcased advanced model tuning and interpretability for predicting a song’s popularity.
* **Classification Pipeline** balanced genre prediction challenges through feature engineering, class rebalancing, and model comparison.

Both workflows follow a **modular, reproducible ML engineering design** — suitable for real-world deployment and showcasing strong practical data science expertise.

> 🚀 *This project reflects skills across supervised learning, data preprocessing, model selection, evaluation, and presentation of insights — all crucial for Data Science and ML Engineer roles.*

---

### 👨‍💻 Author

**Ankit Kothawade**
MSc Data Science — *University of Strathclyde, Glasgow*

* **Email:** [ankitkothawade.ds@gmail.com](mailto:ankitkothawade.ds@gmail.com)
* **LinkedIn:** [linkedin.com/in/ankit-kothawade](https://www.linkedin.com/in/ankit-kothawade)
* **GitHub:** [github.com/ankitkothawade](https://github.com/ankitkothawade)

> Passionate about applying AI, deep learning, and data-driven decision-making in finance, music, and technology.

---

### 🏷️ Keywords

`Machine Learning` · `Regression` · `Classification` · `Scikit-learn` · `Python` · `Feature Engineering` · `Cross Validation` · `RandomForest` · `GradientBoosting` · `XGBoost` · `Data Science Portfolio`

---

### ⭐ Acknowledgment

This repository was developed as part of the **MSc Data Science** coursework and extended into a personal showcase project to highlight practical ML engineering, data preprocessing, and model evaluation proficiency.

> *“Turning raw data into tuned intelligence — one model at a time.”*

---

