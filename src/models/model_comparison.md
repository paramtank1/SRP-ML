# Model Comparison: Directional Stock Prediction

This section compares different models used for next-day
stock direction (Up/Down) prediction.

All models were evaluated using **TimeSeries K-Fold Cross Validation**
to preserve temporal order and avoid future data leakage.

---

## Models Evaluated

1. Logistic Regression (Baseline)
2. Random Forest (Base Features)
3. Random Forest (Enhanced Features)

---

## Cross-Validation Results

| Model | Feature Set | Average CV Accuracy |
|------|------------|--------------------|
| Logistic Regression | Lagged returns, volatility, volume | ~0.504 |
| Random Forest | Base features | ~0.517 |
| Random Forest | Enhanced features (Momentum + Regime) | ~0.519 |

---

## Key Observations

- Random Forest consistently outperformed Logistic Regression,
  indicating the presence of non-linear relationships.
- Feature enhancement led to a measurable improvement in model
  performance, even in a noisy financial dataset.
- The overall accuracy remains close to random guessing, which
  aligns with the weak-form efficiency of stock markets.

---

## Feature Importance (Final Random Forest)

| Feature | Importance |
|-------|------------|
| Return_Lag_5 | High |
| Volume_Change | High |
| Momentum_5 | Medium–High |
| Return_Lag_1 | Medium |
| Rolling_Volatility | Medium |
| Return_Lag_3 | Low |
| Volatility_Regime | Very Low |

Momentum emerged as a strong predictive feature, confirming the
importance of short-term trend information in directional prediction.

---

## Conclusion

While absolute prediction
