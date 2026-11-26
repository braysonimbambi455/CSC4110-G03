# Model Card: Crop Yield Prediction & Classification

## 1. Model Details
- **Developers:** Brayson, Simon, Jacob Chemning'wa (COM/0047/22), Austin, Agnes.
- **Model Date:** November 2026
- **Model Type:** 
  - Regression: Multiple Linear Regression, Regularized Regression.
  - Classification: Random Forest Classifier.
  - Advanced: Neural Networks, Decision Trees.
- **Framework:** Scikit-Learn, Pandas, NumPy (Python 3).

## 2. Intended Use
- **Primary Use:** Predicting agricultural crop yields (log_yield) based on climate and pesticide data, and classifying crop types (Maize, Wheat, etc.) based on environmental features.
- **Out-of-Scope Use:** This model should not be used as the sole decision-maker for real-world farming or financial investments. It is a statistical estimate based on historical data.

## 3. Training & Evaluation Data
- **Data Source:** Historical crop data (`clean_crops.csv`).
- **Features:** Rainfall (`rain_mm`), Average Temperature (`avg_temp`), Pesticides usage (`pesticides_t`), Year, Area frequency.
- **Time Split:** Trained on data up to 2009 (`Year <= 2009`), tested on data from 2010-2013 (`Year >= 2010`).

## 4. Metrics
- **Regression (Jacob's Task 3.1):** Test R² = ___, Test RMSE = ___.
- **Classification (Agnes's Task 4 / Jacob's Task 5):** Test Macro F1-Score = ___. 
- **Validation:** 5-Fold Stratified Cross-Validation used to tune hyperparameters.

## 5. Ethical Considerations & Limitations
- **Data Bias:** The dataset may underrepresent certain crops or specific geographical areas. 
- **Temporal Shift:** Climate patterns change. A model trained on data up to 2009 might not accurately predict extreme weather events in 2024 and beyond.
- **Correlation vs. Causation:** The model finds correlations (e.g., more rain = more yield), but does not prove causation. Applying more pesticides blindly will not guarantee higher yields.
- **Environmental Impact:** Recommending high pesticide use based on model predictions could harm local ecosystems. 

## 6. Caveats
- The model assumes the data fed into the web app is clean and pre-processed exactly like the training data.
