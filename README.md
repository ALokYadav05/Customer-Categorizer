## Customer Categorization ##

This project is a complete end-to-end machine learning application for categorizing customers based on their behavior and demographics. Built from scratch using Python, Flask, and scikit-learn, the app enables users to input customer data and receive a predicted category or cluster in return.

# 🚀 Features: #
📊 Exploratory Data Analysis (EDA) on customer data
⚙️ Data Preprocessing: handled missing values, scaled numerical features
🛠 Feature Engineering: aggregated features, encoded categorical variables
🤖 Unsupervised Learning: used clustering (KMeans) to segment customers
📈 Model Evaluation: used Silhouette Score and visualizations for cluster validation
🧪 Model Building: trained and saved best model using joblib
🌐 Flask App: created a user-friendly HTML form with Bootstrap styling
🧾 Error Handling: catches missing or invalid inputs gracefully
📦 Deployment Ready: local development server setup using Flask

❗Note: Currently, the model is predicting the same cluster for all inputs. Needs improvement in clustering logic and training pipeline.

## Project Structure:
```text
customer_categorizer_project/
├── Deployment/
│   ├── app.py
│   └── templates/
│       ├── index.html
│       └── result.html
├── src/
├── DataPreprocess_&_EDA.ipynb
├── Feature_Engineering.ipynb
├── Classification.ipynb


# Current Limitations: #
* The current model always predicts the same cluster (usually Cluster 0), indicating potential overfitting or poor clustering quality.
* Needs retraining and feature selection adjustments to fix model generalization.

# 📌 Learnings & Contributions: #
* Built the entire pipeline from scratch:
* Data cleaning, encoding, clustering, model saving
* Integrated model into a Flask backend

# 🛠️ Tech Stack: #
Python
scikit-learn
pandas, numpy, matplotlib,Seaborn
Flask (backend)
joblib (model persistence)

## Roadmap / Next Steps
- Rebalance clusters via SMOTE or re-clustering  
- Explore alternative clustering (DBSCAN, GMM)  
- Add model validation metrics and visuals  
- Deploy on production environment using Gunicorn
