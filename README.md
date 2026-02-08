# Stock Return Prediction using Machine Learning

## Project Overview

This project implements an end-to-end machine learning pipeline to
predict the **next-day direction of stock prices (Up/Down)** using
historical market data. The focus is on building a clean, modular,
and industry-aligned workflow rather than chasing unrealistic accuracy.

The project follows proper data preparation, feature engineering,
baseline modeling, time-series validation, and model comparison
techniques suitable for financial datasets.

---

## Technologies Used

- Python 3.13
- Pandas, NumPy
- Scikit-learn
- VS Code
- Git & GitHub

---

## Project Structure

Great question — and honestly, this is a **very good sign** that you’re reviewing your GitHub view critically 👍
Nothing is *wrong* here. What you’re seeing is a **formatting + design choice issue**, not a code or Git problem.

Let me explain it clearly.

---

## Project Structure

```
SRP-ML/
├── data/
│   └── raw/
│       └── Dataset.csv
│
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   └── test_loader.py
│   │
│   ├── features/
│   │   ├── build_features.py
│   │   └── test_features.py
│   │
│   └── models/
│       ├── train_baseline.py
│       ├── train_directional.py
│       ├── train_rf_directional.py
│       ├── model_comparison.md
│       └── final_conclusion.md
│
├── requirements.txt
└── README.md
```


---

## Methodology

1. **Data Preparation**
   - Loaded historical stock price data
   - Normalized timestamps
   - Created next-day return target

2. **Feature Engineering**
   - Lagged returns
   - Rolling volatility
   - Volume change
   - Momentum and volatility regime features

3. **Modeling**
   - Logistic Regression (baseline)
   - Random Forest (non-linear model)

4. **Evaluation**
   - TimeSeries K-Fold Cross Validation
   - Accuracy and feature importance analysis

---

## Results Summary

- Logistic Regression provided a realistic baseline close to random
  guessing, which is expected for short-term stock prediction.
- Random Forest consistently outperformed the baseline by capturing
  non-linear patterns.
- Feature engineering (momentum and regime features) led to measurable
  performance improvement.
- Momentum emerged as one of the most important predictive features.

---

## Key Learning Outcomes

- Importance of proper time-series validation
- Feature engineering has greater impact than model complexity
- Financial markets are noisy and weakly predictable
- Clean project structure improves reproducibility and debugging

---

## How to Run the Project

1. Activate virtual environment
```bash
venv\Scripts\activate
