# LinkedIn post – DRAFT (review and edit before posting yourself)

Replace the two links before posting. Add 2–3 screenshots from `docs/screenshots/`.

---

🚲 How many bikes will a city need at 5 pm on a rainy Wednesday?

For my M.Tech (Data Science & Analytics) course CSE570, I built an end-to-end machine
learning project: a Smart Urban Bike Demand Prediction System.

📊 Data: the real UCI Bike Sharing dataset, 17,379 hourly records from Capital
Bikeshare, Washington D.C. (2011–2012), with weather and calendar information.

🔧 What I built
• Data validation, EDA and feature engineering in Python (pandas, NumPy, Seaborn)
• Regression to predict bikes per hour: Linear, Polynomial, Ridge, Lasso, Elastic Net
• Classification of demand into Low / Medium / High: Logistic Regression, KNN,
  Decision Tree, Random Forest, SVM, Naive Bayes
• Leakage-safe scikit-learn pipelines, a chronological train/validation/test split and
  time-series cross-validation
• An interactive Streamlit dashboard for what-if predictions
• 30 automated tests with pytest

📈 Results on a held-out period (Oct–Dec 2012)
• Regression (Lasso, log target): average error of 42.5 bikes/hour, 73% lower than a
  baseline
• Classification (Logistic Regression): macro F1 0.816, ROC-AUC 0.946

💡 What I learned
• A random split would have hidden a 63% growth in demand. Evaluating on future data
  gives honest numbers.
• Two columns that add up to the target would have produced a "perfect" model. Spotting
  data leakage matters more than picking the fanciest algorithm.
• Accuracy can mislead: a model that always says "High" scored 60% accuracy but only
  0.25 macro F1.
• Even the missing hours in the data had a story: Hurricane Sandy shut the system down
  in Oct 2012.

This is a learning prototype trained on historical data from one city. It is not a live
forecasting service. Next, I would like to try it on Indian city data.

🔗 Code: [GitHub repository link]

#MachineLearning #DataScience #Python #ScikitLearn #Streamlit #Regression #Classification
#MTech #StudentProject #UrbanMobility
