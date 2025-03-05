# Forecasting Compliance Burden: A Data-Driven Approach to Regulatory Cost Estimation

## Date
**2024-12-05**

## Problem Description
This project aims to investigate the estimated compliance costs associated with U.S. federal regulations. The objective is to categorize the regulatory compliance burden into two tiers: **low-cost** and **high-cost** categories. The classification is based on factors such as regulatory restrictiveness, complexity, and industry relevance. By developing a predictive model, the project provides a data-driven framework for understanding the impact of federal regulations on businesses.

## Data Source
- **Public Data Link**: [Dataset](https://drive.google.com/drive/folders/19e-L1y1K2X7JW27ZsErubYz0OFrW06tW?usp=drive_link)

## Data Summary
The dataset includes **21 features** that quantify the restrictiveness, complexity, and structure of federal regulations. Key features include:
- **document_id**: Unique identifier for each regulatory document.
- **year**: Year of publication/enforcement.
- **agency_name**: The regulatory body responsible for the document.
- **restrictions, wordcount, shall, must, prohibited, required**: Metrics assessing the document’s complexity and restrictiveness.
- **sentence_length, acronyms_per_100_sentences**: Indicators of document readability and structure.

## Methodology
### **1. Data Preprocessing**
- **Handling Missing Values**: Numerical columns replaced with median values.
- **Zero-Value Correction**: Adjusted numerical values (e.g., wordcount) to prevent computational errors.

### **2. Feature Engineering**
- **Compliance Score Calculation**: Weighted sum of significant (shall, required) and non-significant terms (may_not, prohibited), normalized by word count.
- **Classification**: Labels assigned as:
  - **Compliant**: Above the mean score.
  - **Non-Compliant**: Below or equal to the mean.

### **3. Data Simplification & Storage**
- Removed intermediate score column post-classification.
- Saved processed dataset for model training.

## Machine Learning Model
### **Model Selection and Training**
- **Algorithm**: **H2O Gradient Boosting (GBM)**
- **Feature Selection**: Evaluated importance of terms like "shall", "must", and document length.
- **Cross-Validation**: Selected the best model based on **AUC** and **MSE** metrics.

### **Model Training Pipeline**
- **Splitting**: Train, Validation, and Test sets.
- **Training on H2O GBM**:
  ```r
  h2o.init()
  model_path <- "Project-Group6-ForecastingComplianceCost-GBM.h2o"
  best_model <- h2o.loadModel(model_path)
  summary(best_model)
  ```

### **Model Evaluation**
- **AUC**: **0.9996**
- **MSE**: **0.0113**

## Explainable AI (XAI) Analysis
### **Feature Importance**
- **Key Influential Features:**
  - **shall** (most important)
  - **wordcount** (document length)
  - **required** (mandatory compliance term)
- **Lesser Influential:**
  - **restrictions_v2, restrictions** (moderate impact)
  - **may_not, must, prohibited** (secondary role)
- **Minimal Influence:**
  - **unique_acronyms, sentence_length** (negligible impact)

### **SHAP Analysis & Prediction Explanation**
- **Top Findings:**
  - **Shorter documents tend to be Non-Compliant**.
  - **Presence of key compliance terms (shall, required) strongly predicts compliance**.
  - **Sentence complexity contributes to classification accuracy**.

## Model Deployment & Future Work
### **Deployment Strategy**
- **H2O Model**: Pre-trained and available for real-time predictions.
- **APIs**: Future implementation for integration into regulatory analysis platforms.

### **Future Enhancements**
- **Dataset Expansion**: Testing model robustness with additional regulations.
- **Feature Exploration**: Evaluating more compliance-related variables.
- **Cross-Domain Applications**: Adapting model for financial & healthcare regulations.

## Presentation
- **Project Overview & Insights**: [RPubs Presentation](https://rpubs.com/aghosh8/1256037)

## Conclusion
The H2O-based predictive model effectively classifies regulatory compliance costs with high accuracy. Explainable AI techniques provided transparency into model decision-making, making this approach viable for regulatory agencies and businesses seeking cost-efficient compliance strategies.

**This project represents a step forward in using AI for regulatory compliance analysis, enabling businesses to better understand and manage compliance burdens.**

