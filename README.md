# Second Look Lending 

> 💡 **Note for the team:** This is just a template. Update the above title with your AI Studio Challenge Project name. Remove all guidance notes and example text in this template and populate this README with your own content. You can work on this README throughout AI Studio, and get feedback from your AI Studio Coach and Challenge Advisor before finalizing it.  

---

### 👥 **Team Members**

| Name             | GitHub Handle | Contribution                                                             |
|------------------|---------------|--------------------------------------------------------------------------|
| Christopher Chow | @CC-001-CC    | Data Squashing, Model Training                                           |
| James Shaw       | @jimxshaw     | Data Squashing, Data pipeline integration, Model Building                |
| Lauren Kim       | @laurenyk     | EDA                                                                      |
| Nitya Konduru    | @nityak8      | Data Preprocessing                                                       |
| Kelly Garcia     | @kelly2377    | Data Exploration, Data Cleaning                                          |
| Angela Mashuryan | @angelam118   | Data Exploration                                                         |
| Samiha Haque     | @samiha3231   | Data Exploration, Feature Engineering, EDA, Model Training               |
---

## 🎯 **Project Highlights**

- Analyzed synthetic relational data representing approximately 1 million loan applications.
- Developed a machine learning workflow to estimate applicants' probability of default.
- Used a chronological split based on `WEEK_NUM` to reduce data leakage and evaluate performance on future applications.
- Expanded the original feature set to 106 features through feature engineering and aggregation.
- Investigated differences in default risk between thin-file applicants and applicants with established credit histories.
- Developed a Streamlit application that demonstrates predicted default risk, approval decisions, and expected profit using synthetic applicant profiles.
- Evaluated lending decisions using both predictive performance and business considerations.
---

## 👩🏽‍💻 **Setup and Installation**

**Provide step-by-step instructions so someone else can run your code and reproduce your results. Depending on your setup, include:**

* How to clone the repository
* How to install dependencies
* How to set up the environment
* How to access the dataset(s)
* How to run the notebook or scripts

### 1. Clone the Repository

```bash
git clone https://github.com/Break-Through-Tech/American-Express-1B-second-look-lending.git
cd American-Express-1B-second-look-lending
```

### 2. Install the Required Packages

Install the project dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Get the Data

The dataset is not included in this repository because of its large file size. Download `synth_home_credit.zip` from the [shared project Google Drive folder](https://drive.google.com/drive/folders/1bJpa87n6HCwRR3Pvr5xqocNZrGo7dufT), then extract the files into your local data directory.

### 4. Run the Notebooks

Open Jupyter Notebook or Google Colab and run the notebooks located in the `notebooks` folder.

The main exploratory data analysis and feature-engineering notebook is:

```text
notebooks/feature_engineering_EDA.ipynb
```

### 5. Run the Streamlit Application

Launch the demonstration application with:

```bash
streamlit run app/app.py
```
---

## 🏗️ **Project Overview**

Our host company, American Express, challenged the team to explore second-look lending. The project focuses on applicants who may have been initially denied credit, including thin-file applicants with limited credit histories.

The primary objectives are to:

1. Estimate each applicant's probability of default.
2. Compare machine learning models for default-risk prediction.
3. Develop an approval policy that considers both expected profit and default risk.
4. Evaluate model outcomes for thin-file and established-credit applicants.
5. Explore how additional information can help identify creditworthy applicants who may otherwise be denied.

The project uses an assumed gain of $2,000 for a repaid loan and a loss of $8,000 for a defaulted loan. These values produce a break-even default-probability threshold of 20%.

A successful solution could support more informed lending decisions while examining opportunities to improve access for applicants with limited credit histories.

---

## 📊 **Data Exploration**

The project uses reproducible synthetic consumer-lending data. 

The data include:

- Approximately 1 million training applications
- Approximately 200,000 out-of-time test applications
- Numerical, categorical, date, and relational features
- Multiple related tables connected through `case_id`
- A binary target where `1` represents default and `0` represents repayment
- An imbalanced target distribution, with approximately 19% defaults and 81% repayments

### Data Preparation

The data preparation and feature-engineering process includes:

- Joining related datasets using `case_id`
- Aggregating previous-loan, tax, personal, and credit-bureau records
- Creating count, mean, minimum, maximum, sum, and recency features
- Creating affordability and leverage ratios
- Creating credit-behavior and overdue-payment features
- Adding missingness and data-coverage indicators
- Identifying thin-file applicants
- Examining hashed categorical variables, including education
- Encoding categorical variables
- Identifying highly correlated and redundant features
- Handling missing and extreme values using training-data thresholds

### Chronological Split

The data were divided chronologically instead of randomly:

- **Training set:** Weeks 0 through 79, containing 869,559 applications
- **Validation set:** Weeks 80 through 91, containing 130,441 applications

This split more closely represents predicting outcomes for future applicants and reduces the risk of data leakage.

### EDA Findings

- Thin-file applicants had a default rate of approximately **26.51%**.
- Applicants with established credit histories had a default rate of approximately **16.42%**.
- Feature engineering increased the dataset from 45 to 85 features.
- One-hot encoding increased the modeling dataset to 104 columns.
- Several features contained substantial missingness.
- Some variables were highly or perfectly correlated and were flagged for removal.
- Individual features showed weak linear correlations with the target, supporting the evaluation of models that can capture nonlinear relationships.

---

## 🧠 **Model Development**

Currently in progress of building models and analyzing various ML algorithms. 

---

## 📈 **Results & Key Findings**

To be completed after modeling is finished.

---

## 🚀 **Next Steps**
- Finalize models for thin-files and regular applicants 
- Add final performance metrics and visualizations to the README.
- Improve the Streamlit application's usability and documentation.
- Evaluate the limitations and fairness implications of the final model.

---

## 📝 **License**

To be done after speaking with the Challenge Advisor and Coach. 

---

## 🙏 **Acknowledgements** (Optional but encouraged)

Thank you to our Challenge Advisor Saurabh Gupta, AI Studio Coach Swagath Babu, and American Express.
