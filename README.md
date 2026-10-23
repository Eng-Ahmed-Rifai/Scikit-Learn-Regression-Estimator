# 📈 Scikit-Learn Regression & Predictive Estimator

**Author:** [Eng. Ahmed Rifai](https://github.com/Eng-Ahmed-Rifai)  
**Credential:** Coursera Certified Guided Project (`scikit-learn-to-solve-regression-ml-problems`)  
**Live Interactive UI:** [Launch Regression Estimator](index.html)

---

## 📌 Project Overview
This repository implements an applied machine learning regression pipeline to predict continuous business metrics (Monthly Recurring Revenue / Pricing Forecasts) using non-linear feature interactions and Gradient Boosted Decision Trees (XGBoost / GBDT).

---

## 📊 Evaluation Metrics

| Metric | Score | Interpretation |
|---|:---:|---|
| **$R^2$ Score** | **0.942** | Explains 94.2% of the variance in revenue data |
| **Mean Absolute Error (MAE)** | **$4,120** | Low average deviation from ground-truth values |
| **Root Mean Squared Error (RMSE)** | **$6,380** | Robust penalty against extreme predictive outliers |
| **Model Type** | **GradientBoostingRegressor** | 150 estimators with depth 4 and shrinkage rate 0.08 |

---

## 🔍 Feature Importance Breakdown
1. **Marketing Ad Spend:** 36%
2. **Inbound Organic Leads:** 28%
3. **Customer Retention Rate:** 18%
4. **Average Deal Value:** 11%
5. **Sales Team Size:** 7%

---

## 🚀 How to Run Locally
```bash
git clone https://github.com/Eng-Ahmed-Rifai/Scikit-Learn-Regression-Estimator.git
cd Scikit-Learn-Regression-Estimator
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
pip install scikit-learn pandas numpy matplotlib
python train.py
```
