#  Airbnb Rental Price Prediction in Melbourne

This project develops and evaluates several machine learning models to accurately predict the rental price of Airbnb listings in **Melbourne, Australia**. The goal is to help hosts optimize their pricing strategy and provide guests with better transparency on pricing trends.

The **Gradient Boosting Regressor** emerged as the best-performing model, achieving an **$R^{2}$ score of 0.9796** and a low **Root Mean Squared Error (RMSE) of 11.88** on the test set.

***

## Project Goal

The primary objective is to create a machine learning pipeline that can accurately estimate the price of Airbnb listings based on a variety of features such as location, property type, number of bedrooms, amenities, and reviews[cite: 8, 11].

***

## Dataset and Features

The dataset was sourced from **Inside Airbnb** (`listings.csv` for Melbourne)[cite: 15, 16].

| Detail | Value |
| :--- | :--- |
| **Source** | http://insideairbnb.com/get-the-data/ [cite: 15] |
| **Initial Samples** | ~20,000 [cite: 17] |
| **Cleaned Samples** | 17,106 (after cleaning and outlier removal) |
| **Target Variable** | `price` (in AUD) [cite: 25] |
| **Key Features** | `property_type`, `room_type`, `bedrooms`, `bathrooms`, `beds`, `amenities`, `reviews_count`, `neighbourhood`, `latitude`, `longitude` [cite: 19, 20, 21, 22, 23, 24] |

***

##  Methodology and Workflow

The project followed a standard machine learning workflow:

1.  **Data Cleaning & Preprocessing**: Handled missing values (using median imputation for numerics) and removed outliers using the Interquartile Range (IQR) method on the `price` column[cite: 42].
    * *Initial dataset shape:* (18539, 21)
    * *Shape after outlier removal:* (17106, 21)
2.  **Feature Engineering**: Categorical features like `neighbourhood` and `room_type` were encoded using **Label Encoding**[cite: 37]. Non-numeric and problematic columns were dropped.
3.  **Train/Test Split**: The dataset was split into an **80% training** set (13,684 samples) and a **20% testing** set (3,422 samples)[cite: 43].
4. **Feature Scaling**: Features were standardized using `StandardScaler` to prepare them for model training[cite: 40].
5. **Model Training**: Several regression models were trained, including Linear Models and Ensemble Methods (Random Forest and Gradient Boosting)[cite: 44].
6. **Model Evaluation**: Performance was assessed using three key metrics: **RMSE, MAE, and $R^{2}$**[cite: 49, 50, 51, 52].
7. **Persistence**: The final, best-performing model (`Gradient Boosting`) was saved using the `pickle` library for deployment[cite: 66].

***

##  Results and Model Comparison

The models were evaluated on the test set. The ensemble methods demonstrated superior performance compared to the linear models[cite: 60].

### Evaluation Metrics

| Metric | Description | Goal |
| :--- | :--- | :--- |
| **Root Mean Squared Error (RMSE)** | Average magnitude of errors (typical error size in predicted price)[cite: 50]. | Lower is better. |
| **Mean Absolute Error (MAE)** | Average absolute difference between predicted and actual values[cite: 51]. | Lower is better. |
| **R-squared ($R^{2}$)** | Proportion of variance in the target variable explained by the features[cite: 52]. | Closer to 1.0 is better[cite: 53]. |

### Performance Table

| Model | RMSE | MAE | $R^{2}$ |
| :--- | :--- | :--- | :--- |
| Gradient Boosting | **11.8810** | **9.1136** | **0.9796** |
| Random Forest | 14.3543 | 9.3705 | 0.9702 |
| Linear Regression | 23.2691 | 16.7375 | 0.9217 |
| Ridge ($\alpha=100.0$) | 23.3189 | 16.7813 | 0.9213 |
| Lasso ($\alpha=1.0$) | 23.4795 | 17.0245 | 0.9203 |

**Conclusion**: The **Gradient Boosting** model is the best performer, explaining nearly **98%** of the price variance[cite: 60, 62].

***

##  Deployment and Future Work

The final model and necessary artifacts (`best_model.pkl`, `scaler.pkl`, `feature_names.pkl`, `le_dict.pkl`) are saved for future deployment and integration[cite: 66]. A simple prediction interface was built using the **Gradio** library.

### Future Enhancements [cite: 65]
* Explore deep learning approaches.
* Incorporate external data sources (e.g., local events, seasonal trends).
* Develop a user-friendly interface for real-time price predictions.

***

##  Team Details

| Name | SRN |
| :--- | :--- |
| Cherukuri Venkata Kartik | PES2UG23CS148 |
| Gavini Prithvi Ananya | PES2UG23CS198 |
| Fairly Sorathiya | PES2UG23CS189 |
