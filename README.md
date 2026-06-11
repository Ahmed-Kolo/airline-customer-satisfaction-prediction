```markdown
# Airline Customer Satisfaction Prediction

## Project Overview
This project aims to develop and evaluate machine learning models to predict customer satisfaction based on various airline passenger data. By analyzing factors such as flight experience, service quality, and demographics, we seek to identify key drivers of satisfaction and build predictive models that can assist the airline in improving customer experience and retention. The project involved data preprocessing, training and optimizing a Decision Tree Classifier, and comparing its performance against a Logistic Regression model.

## Data Preprocessing

### Initial Data Loading and Inspection
The project began by loading the `Invistico_Airline.csv` dataset. Initial inspection revealed numerical and categorical features, along with a target variable `satisfaction`.

### Handling Missing Values
*   **Arrival Delay in Minutes**: Missing values in this column were imputed with the mean of the column to preserve data integrity and prevent loss of rows.

### Encoding Categorical Features
*   **Categorical Features**: All `object` type columns (e.g., `Customer Type`, `Type of Travel`, `Class`) were one-hot encoded to convert them into a numerical format suitable for machine learning models. The `drop_first=True` argument was used to avoid multicollinearity.
*   **Target Variable (`satisfaction`)**: The `satisfaction` column, originally containing 'satisfied' and 'dissatisfied', was mapped to numerical values (1 for 'satisfied', 0 for 'dissatisfied').

### Problem Resolution: Target Variable Encoding Error
During initial preprocessing, an error led to the target variable `satisfaction` having only one unique class. This was traced back to an incorrect mapping where 'neutral or dissatisfied' was implicitly dropped, resulting in only 'satisfied' instances remaining. The issue was resolved by reloading the original dataset and applying the correct mapping: `{'satisfied': 1, 'dissatisfied': 0}`, ensuring both classes were correctly preserved. This corrected dataset, `df_processed`, was then used for subsequent steps, ensuring a balanced representation of both satisfied and dissatisfied customers.

### Data Splitting
The processed data was split into training (80%) and testing (20%) sets using `train_test_split`, with `stratify=y` to maintain the proportion of satisfied and dissatisfied customers in both sets, crucial for robust model evaluation.

## 

### Decision Tree Classifier
A Decision Tree Classifier was chosen for its interpretability and ability to handle complex relationships. To optimize its performance, `GridSearchCV` was employed to tune hyperparameters, specifically `max_depth`, `min_samples_split`, and `min_samples_leaf`. The model was trained to maximize the F1-score.

*   **Best Hyperparameters Found:**
    *   `max_depth`: `None` (unlimited depth)
    *   `min_samples_leaf`: `2`
    *   `min_samples_split`: `10`

### Logistic Regression Model
As a baseline for comparison, a Logistic Regression model was also trained on the same preprocessed data. Initial attempts with the `saga` solver faced convergence issues, even after increasing `max_iter`. This was resolved by switching to the `liblinear` solver, which proved more robust for the dataset.

## Model Evaluation

### Decision Tree Classifier Evaluation
After retraining with the corrected target variable, the optimized Decision Tree Classifier yielded strong performance metrics on the test set:
*   **Best Hyperparameters:** `{'max_depth': None, 'min_samples_leaf': 2, 'min_samples_split': 10}`
*   **F1-score for 'Satisfied' class (Class 1):** `0.9426`
*   **Confusion Matrix:**
    ```
    [[11059   700]
     [  920 13297]]
    ```
    *   **True Positives (TP):** 13,297 (Correctly identified satisfied customers)
    *   **True Negatives (TN):** 11,059 (Correctly identified dissatisfied customers)
    *   **False Positives (FP):** 700 (Customers incorrectly predicted as satisfied)
    *   **False Negatives (FN):** 920 (Customers incorrectly predicted as dissatisfied)

### Logistic Regression Model Evaluation
*   **F1-score for 'Satisfied' class (Class 1):** `0.8436`
*   **Confusion Matrix:**
    ```
    [[ 9543  2216]
     [ 2230 11987]]
    ```
    *   **True Positives (TP):** 11,987
    *   **True Negatives (TN):** 9,543
    *   **False Positives (FP):** 2,216
    *   **False Negatives (FN):** 2,230

## Model Comparison

This section compares the performance, interpretability, and business implications of the Decision Tree and Logistic Regression models for predicting customer satisfaction.

### Performance Metrics

**Decision Tree Model:**
*   **F1-score for 'Satisfied' class:** `0.9426`
*   **Confusion Matrix:**
    ```
    [[11059   700]
     [  920 13297]]
    ```
    *   **True Positives (TP):** 13,297
    *   **True Negatives (TN):** 11,059
    *   **False Positives (FP):** 700
    *   **False Negatives (FN):** 920

**Logistic Regression Model:**
*   **F1-score for 'Satisfied' class:** `0.8436`
*   **Confusion Matrix:**
    ```
    [[ 9543  2216]
     [ 2230 11987]]
    ```
    *   **True Positives (TP):** 11,987
    *   **True Negatives (TN):** 9,543
    *   **False Positives (FP):** 2,216
    *   **False Negatives (FN):** 2,230

**Comparison:**
The Decision Tree model significantly outperforms the Logistic Regression model in terms of F1-score for the 'Satisfied' class (0.9426 vs. 0.8436). This indicates that the Decision Tree model has a better balance of precision and recall in identifying satisfied customers. The Decision Tree also shows a lower number of False Positives and False Negatives, meaning it makes fewer classification errors overall compared to Logistic Regression.

### Interpretability

**Decision Tree:**
*   **High Interpretability:** Decision Trees are inherently interpretable. The visualization of the tree structure (as shown in a previous step) clearly illustrates the decision rules and pathways the model uses to classify customers. This makes it easy to explain to non-technical stakeholders why a particular customer is predicted as 'satisfied' or 'dissatisfied' by tracing the path through the tree.
*   **Rule-based:** The decision process is based on a series of 'if-then-else' rules, which aligns well with human reasoning.

**Logistic Regression:**
*   **Moderate Interpretability:** Logistic Regression coefficients indicate the direction and strength of the relationship between features and the log-odds of the target variable. While numerical coefficients can be understood, they are less intuitive for non-technical users than direct decision rules. Explaining the impact of multiple features simultaneously can be complex.

### Handling of Relationships

**Decision Tree:**
*   **Non-linear Relationships:** Decision Trees can naturally capture complex non-linear relationships and interactions between features without requiring explicit feature engineering for such interactions. This flexibility likely contributes to its higher performance in this dataset.

**Logistic Regression:**
*   **Linear Relationships:** Logistic Regression primarily models linear relationships between features and the log-odds of the target. While it can incorporate non-linearities through polynomial features or interaction terms, it requires explicit definition of these, which was not done in this implementation. This limitation may explain its lower performance compared to the Decision Tree.

### Business Actionability

**Decision Tree:**
*   **High Actionability:** The feature importances derived from the Decision Tree provide clear insights into the most influential factors for customer satisfaction (e.g., Inflight entertainment, Seat comfort). These insights are directly actionable for business strategies, allowing the airline to prioritize improvements in areas that have the greatest impact on customer satisfaction. For example, knowing that 'Inflight entertainment' is the top driver, the airline can invest more in enhancing this service.

**Logistic Regression:**
*   **Moderate Actionability:** While the coefficients of Logistic Regression can indicate the importance of features, their interpretation is less straightforward for direct business action compared to the hierarchical rules of a Decision Tree. It helps understand positive or negative influences, but doesn't offer clear decision paths for customer segmentation or service improvement strategies.

### Trade-offs and Conclusion

**Decision Tree:**
*   **Pros:** Higher predictive performance, excellent interpretability, handles non-linear relationships well, and provides clear, actionable insights through feature importance and decision rules.
*   **Cons:** Can be prone to overfitting if not properly pruned (though GridSearchCV addressed this), and slight changes in data can lead to a very different tree structure.

**Logistic Regression:**
*   **Pros:** Simpler model, computationally less intensive, good baseline model, and provides a probabilistic output.
*   **Cons:** Lower predictive performance in this scenario, assumes linear relationships, and less intuitive interpretability for non-technical audiences compared to Decision Trees.

**Conclusion for Stakeholders:**
Given the objective of predicting customer satisfaction and deriving actionable insights, the **Decision Tree model is the preferred choice**. It demonstrates superior predictive accuracy (F1-score of 0.9426) and offers unparalleled interpretability, allowing us to pinpoint exact reasons for customer satisfaction or dissatisfaction. The clear feature importance ranking from the Decision Tree provides direct guidance on where the airline should focus its efforts to improve customer experience, such as prioritizing **inflight entertainment** and **seat comfort**. While Logistic Regression provides a decent baseline, its performance and interpretability are not as well-suited for extracting the granular, actionable insights required for this business problem.

## Key Findings and Actionable Insights

Based on the analysis and model comparison, several key findings and actionable insights emerge:

*   **Decision Tree Superiority:** The Decision Tree model significantly outperformed the Logistic Regression model with an F1-score of `0.9426` for the 'Satisfied' class, demonstrating its superior ability to accurately predict customer satisfaction. Its high interpretability allows for a clear understanding of the decision-making process.

*   **Top Drivers of Satisfaction:** The feature importance analysis from the Decision Tree identified the following as the most influential factors:
    1.  **Inflight entertainment:** `0.418269`
    2.  **Seat comfort:** `0.183992`
    3.  **Ease of Online booking:** `0.062728`
    4.  **Flight Distance:** `0.036500`
    5.  **Departure/Arrival time convenient:** `0.029797`

*   **Actionable Insights:**
    *   **Prioritize Inflight Entertainment and Seat Comfort:** The airline should focus significant efforts and investment on enhancing the quality and variety of inflight entertainment and improving seat comfort. These are the top two drivers directly impacting customer satisfaction.
    *   **Optimize Online Booking Experience:** Improving the ease of online booking is another critical area that can yield substantial positive impacts on customer satisfaction.
    *   **Targeted Interventions:** Leveraging the interpretable rules of the Decision Tree, the airline can develop targeted interventions for specific customer segments. For example, understanding what combination of factors leads to dissatisfaction can help design proactive measures to address potential issues before they escalate.
    *   **Monitor Core Services:** Continuously monitor and improve services related to flight distance and departure/arrival time convenience, as these also play a role in the overall customer experience.

These insights provide a data-driven roadmap for the airline to enhance customer experience, improve retention, and foster greater satisfaction among its passengers.
```
