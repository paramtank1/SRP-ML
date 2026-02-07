# Model Comparison: Directional Stock Prediction

This section compares the performance of two models used for
next-day stock direction (Up/Down) prediction.

---

## Models Evaluated

- Logistic Regression (Baseline)
- Random Forest Classifier

Both models were evaluated using **TimeSeries K-Fold Cross Validation**
to avoid future data leakage.

---

## Cross-Validation Results

| Model               | Average CV Accuracy |
|--------------------|--------------------|
| Logistic Regression | ~0.504 |
| Random Forest       | ~0.517 |

---

## Observations

- Random Forest outperforms Logistic Regression by capturing
  non-linear relationships in the data.
- Logistic Regression serves as a strong baseline but is limited
  to linear decision boundaries.
- Performance gains are modest, which is expected due to the
  noisy and weakly predictable nature of financial markets.

---

## Feature Importance (Random Forest)

| Feature              | Importance |
|----------------------|------------|
| Return_Lag_5         | High |
| Volume_Change        | High |
| Return_Lag_1         | Medium |
| Rolling_Volatility   | Medium |
| Return_Lag_3         | Low |

This indicates that medium-term momentum and volume dynamics
play a more significant role in directional prediction.

---

## Conclusion

Although prediction accuracy remains close to random guessing,
the Random Forest model demonstrates improved stability and
better utilization of engineered features, making it more
suitable for further experimentation and feature enhancement.
