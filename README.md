🚀 Job Role Prediction Project

📖 Overview

This project is part of my Infosys Internship and aims to predict suitable job roles for candidates based on their educational background, certifications, and skills using Machine Learning.
The work is divided into structured milestones, ensuring a systematic and professional approach from data preparation to final deployment.

🎯 Objectives

🟢 Predict job roles from resumes and candidate profiles

🟢 Build an end-to-end ML pipeline (data cleaning → feature engineering → model training → deployment)

🟢 Demonstrate strong knowledge of EDA, feature engineering, ML models, and deployment tools

🏆 Milestones
✅ ## **Milestone 1 – Data Cleaning & Exploratory Data Analysis**

📓 Notebook: Infosys(M-1).ipynb

🔹 Collected and integrated datasets (Education, Certifications, Skills, Job Roles)

🔹 Cleaned and preprocessed the raw dataset

🔹 Handled missing values and standardized text fields

🔹 Performed Exploratory Data Analysis (EDA) with meaningful visualizations

📌 Outputs:

🟡 Cleaned dataset ready for ML

🟡 EDA report notebook

Perfect 👍 — here’s a **well-formatted and professional explanation** of your entire process, written in a way that’s ideal for uploading to **GitHub** (for your Milestone or preprocessing notebook).
You can paste this directly into your `README.md`, `Milestone_Report.md`, or as a documentation section in your project folder.
It clearly explains every stage step-by-step, including purpose, methods, and outcomes.



 🧠 ## **Milestone-2: Data Preprocessing and Visualization**

 🔍 **Overview**

This milestone focuses on **cleaning, transforming, and visualizing** the *Edu2Job Dataset* to prepare it for machine learning model development.
The process ensures the data is consistent, well-structured, and free from issues such as missing values, outliers, or format inconsistencies.


 ⚙️ **Steps Followed**

 **1️⃣ Load the Dataset**

* The dataset `Edu2Job_dataset_.csv` is loaded using **Pandas**.
* Initial inspection includes displaying:

  * Dataset shape
  * Column names
  * Sample data
* Verifies that the file is correctly located and accessible.

   python
df = pd.read_csv(file_path)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

**2️⃣ Handling Missing Values**

Missing data can reduce model accuracy if not managed properly.

* **Numeric Columns** → filled with the **mean** of that column.
* **Categorical Columns** → filled with the **most frequent value (mode)** or `"Unknown"` if mode doesn’t exist.
* This approach keeps dataset size consistent and prevents errors during model training.

 python
num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(exclude=[np.number]).columns


✅ *All missing values handled safely — no warnings.*

**3️⃣ Handling Outliers (IQR Method)**

Outliers can distort statistical analysis and model performance.
Used the **Interquartile Range (IQR)** method:

* Values below **Q1 − 1.5 × IQR** or above **Q3 + 1.5 × IQR** are **capped** to boundary values.
* This preserves data shape but prevents extreme deviations.

✅ *Outliers successfully capped.*


 **4️⃣ Label Encoding for Categorical Data**

Machine-learning models require numeric input.

* Applied **LabelEncoder** to convert text data into numeric form.
* Stored encoders for decoding later if needed.

Example:

| Degree | Encoded |
| ------ | ------- |
| B.Tech | 0       |
| M.Tech | 1       |
| MBA    | 2       |

✅ *All categorical features encoded numerically.*



 **5️⃣ Feature Scaling (Standardization)**

To ensure all numeric features are on a similar scale:

* Used **StandardScaler** → transforms features to have **mean = 0**, **std = 1**.
* Prevents large-scale features (like GPA or Experience) from dominating the model.

✅ *Numeric features standardized successfully.*



 **6️⃣ Feature Selection**

Separated data into:

* **X:** Input features (Degree, Major, GPA, Certifications, Skills, Industry, Experience Level)
* **y:** Target variable (`Job Role`)

This prepares the dataset for supervised learning.

✅ *Features and target variable successfully defined.*


 **7️⃣ Train-Test Split**

To evaluate model performance objectively:

* Split the dataset → **70 % training** and **30 % testing**.
* Used `stratify=y` to keep job-role category proportions consistent.
* Random seed fixed (`random_state = 42`) for reproducibility.

✅ *Train/Test split completed successfully.*

 **8️⃣ Save Preprocessed Data**

Two CSV files were exported for transparency:

* **Before preprocessing:** `Edu2Job_Before_Preprocessing.csv`
* **After preprocessing:**  `Edu2Job_After_Preprocessing.csv`

✅ *Files saved for documentation and reproducibility.*



 **9️⃣ Visualizations**

To explore feature relationships, several plots were generated using **Seaborn** and **Matplotlib**:

| Plot                                        | Description                                      |
| ------------------------------------------- | ------------------------------------------------ |
| **Job Role Distribution**                   | Frequency of each job role                       |
| **Degree vs Job Role**                      | How degrees relate to job placement              |
| **GPA vs Job Role (Boxplot)**               | GPA variation across roles                       |
| **Certifications vs Job Role (Strip Plot)** | Certification influence                          |
| **Industry Distribution**                   | Count of samples in each industry                |
| **Experience Level vs Job Role**            | Relationship between experience and job type     |
| **Correlation Heatmap**                     | Strength of relationships among numeric features |

✅ *Visualization completed successfully — helped reveal important insights.*



 📊 **Final Results**

| Step                     | Status |
| ------------------------ | ------ |
| Dataset Loaded           | ✅      |
| Missing Values Handled   | ✅      |
| Outliers Capped          | ✅      |
| Label Encoding           | ✅      |
| Feature Scaling          | ✅      |
| Train-Test Split         | ✅      |
| Visualizations           | ✅      |
| Preprocessed Files Saved | ✅      |

**Dataset Summary**

* Rows: 1000 Columns: 8
* Target: *Job Role*
* Minimum accuracy potential: High (clean & standardized data)

 🧾 Conclusion

This milestone establishes a **clean, well-structured dataset** ready for model building.
By handling missing data, normalizing scales, encoding categories, and visualizing relationships, we created a solid foundation for predictive modeling — specifically, predicting job roles based on educational background and skillset.

Next step → proceed to **model training and evaluation** (Milestone-3).

Edu2Job_Project/
│
├── data/
│   ├── Edu2Job_dataset_.csv
│   ├── Edu2Job_Before_Preprocessing.csv
│   └── Edu2Job_After_Preprocessing.csv
│
├── notebooks/
│   └── Edu2Job_Preprocessing.ipynb
│
├── scripts/
│   └── Edu2Job_Preprocessing.py
│
├── visuals/
│   └── output_plots/
│
├── README.md
└── requirements.txt

Absolutely ✅
Here is the **Milestone-3: Ensembling Machine Learning Models** documentation in the *same structure, tone, and formatting* as your Milestone-2 write-up — ready to upload into GitHub.

🧠 ## **Milestone-3: Ensembling Machine Learning Models**

🎯 Model Training, Evaluation & Performance Comparison

 🔍 Overview

This milestone focuses on applying various supervised machine learning classification algorithms on the preprocessed Edu2Job dataset.
Each model is trained, evaluated, and compared to determine the most accurate job-role prediction system.

The goal is to identify which ML technique best captures the relationship between **education + skills → job outcomes**.

 🤖 Machine Learning Algorithms Applied

| Sl.No | Model Name                   | Type              | Performance Goal           |
| ----- | ---------------------------- | ----------------- | -------------------------- |
| 1     | Logistic Regression          | Linear Classifier | Strong baseline accuracy   |
| 2     | K-Nearest Neighbors (KNN)    | Distance-Based    | Local similarity detection |
| 3     | Decision Tree                | Tree-Based        | Interpretability           |
| 4     | Support Vector Machine (SVM) | Kernel-Based      | High-margin classification |
| 5     | Random Forest                | Bagging Ensemble  | Reduces overfitting        |
| 6     | AdaBoost                     | Boosting          | Learns from weak learners  |
| 7     | Gradient Boosting            | Boosting          | Handles complex patterns   |
| 8     | XGBoost                      | Advanced Boosting | Better speed & accuracy    |

✅ All models successfully executed without warnings.
✅ Hyperparameters kept default for baseline comparison.

 ⚙️ Workflow / Procedure

 1️⃣ Load Preprocessed Dataset

Data imported from:

✔ Edu2Job_After_Preprocessing.csv
✔ Scaled + Encoded features ensured compatibility with ML models

 2️⃣ Feature Selection

* **X → Independent Features** (Education, Skills, Industry…)
* **y → Job Role (Target class)**

 3️⃣ Train-Test Split

📊 70% Training | 30% Testing
`random_state=42` → ensures reproducibility
`stratify=y` → maintains class proportion

 4️⃣ Model Training & Testing

Each classifier was:

✅ Fit on training dataset
✅ Used to predict on test dataset
✅ Evaluated using classification metrics

 📊 Evaluation Metrics

✔ Accuracy Score
✔ Confusion Matrix
✔ Classification Report
(Precision, Recall, F1-Score)

A results table was created to rank models by performance.

 📈 Visualizations

A bar graph plot was generated to compare accuracy scores across models.

| Visualization                 | Purpose                               |
| ----------------------------- | ------------------------------------- |
| Accuracy Comparison Bar Chart | Identifies top-performing model       |
| Confusion Matrices (Optional) | Understand misclassification patterns |

✅ Accuracy visualization clearly highlights boosting models performing better.

 🏆 Results Summary

| Model                                                  | Performance Notes                        |
| ------------------------------------------------------ | ---------------------------------------- |
| Boosting Models (XGBoost, Gradient Boosting, AdaBoost) | ✅ Best results                           |
| Random Forest                                          | 🔹 Good accuracy                         |
| SVM                                                    | Moderate accuracy depending on kernel    |
| KNN                                                    | Performance depends on nearest neighbors |
| Logistic Regression                                    | Serves as a baseline                     |

📌 **Highest Accuracy Model → XGBoost Classifier**
Strong generalization + robust handling of features.

 📁 Files Generated / Updated

| File Name                       | Description                |
| ------------------------------- | -------------------------- |
| `Edu2Job_Model_Training.py`     | Source code for all models |
| `Model_Accuracy_Comparison.png` | Performance bar chart      |
| `Model_Results.csv`             | Accuracy results saved     |

 ✅ Final Status

| Task                  | Status |
| --------------------- | :----: |
| Dataset Imported      |    ✅   |
| Models Trained        |    ✅   |
| Comparisons Completed |    ✅   |
| Visuals Generated     |    ✅   |
| Best Model Identified |    ✅   |



 🧾 Conclusion

This milestone validates multiple ML approaches to classify job roles from educational and experience-based features. Boosting techniques — especially **XGBoost** — showed superior learning and higher predictive capability.






