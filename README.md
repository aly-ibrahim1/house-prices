# House Prices Prediction (Kaggle Competition)

### Overview
This project tackles the classic Kaggle House Prices: Advanced Regression Techniques competition.
The goal is to predict home sale prices in Ames, Iowa based on 80+ explanatory features ranging from lot size and year built to neighborhood and amenities.

I developed an **end-to-end machine learning pipeline** with extensive feature engineering and hyperparameter tuning, **achieving a public leaderboard RMSLE of 0.12994**


### Methodology

The project was designed to mimic a production-ready workflow, broken into stages:

1.	Exploratory Data Analysis (EDA)
	  •	Identified outliers, visualized feature distributions, examined missingness patterns.
	  •	Created a missingness report to guide imputation strategy.
   
2.	Preprocessing
  •	Type casting (categorical, ordinal, numeric).
  •	Consistent handling of missing values using semantic rules (e.g., “NA = no garage”).
  •	Encoding categorical variables (ordinal mappings, one-hot, target encoding).
  •	Baseline model evaluation with Ridge and XGBoost.
   
3.	Feature Engineering
  •	Interaction features (e.g., TotalSF, TotalBaths, BathsPerRoom).
  •	KMeans clustering to group houses by size/layout → added cluster labels as features.
  •	PCA on structural features → extracted principal components + interaction terms.
  •	Mutual Information filtering to drop uninformative features.
   
4.	Hyperparameter Tuning
  •	Used Optuna to optimize XGBoost hyperparameters with 5-fold cross-validation.
  •	Tuned tree depth, learning rate, number of estimators, regularization, and sampling.
   
5.	Model Training & Evaluation
  •	Trained final XGBoost regressor on log-transformed target.
  •	Evaluated performance with RMSLE and MAE:
  •	Cross-validated RMSLE: 0.11666
  •	Cross-validated MAE: ~14,800 USD
  •	Kaggle Public Leaderboard RMSLE: 0.12994



## Results

| Stage                    | RMSLE   | MAE ($)   |
|---------------------------|---------|-----------|
| Baseline after preprocessing (XGB default)    | 0.136   | 17,176    |
| Final Tuned XGB           | **0.116** | **14,800** |
| Kaggle Submission         | **0.1299** | — (hidden labels) |

Demonstrated strong generalization from CV to Kaggle test data.


## Repository Structure
- [src/](src) → Core functions (preprocessing, feature_eng, training, utils)
- [notebooks/](notebooks) → EDA, preprocessing, feature engineering, tuning
- [requirements.txt](requirements.txt) → Dependencies
- [README.md](README.md) → Project overview (this file)
- [.gitignore](.gitignore)


### Branches:
	•	main → Final end-to-end pipeline.
	•	preprocessing → Data cleaning and encoding steps.
	•	feature-eng → Interaction, clustering, PCA experiments.
	•	hyperparam-tuning → Optuna experiments and results.


### Skills Demonstrated
	•	Data preprocessing: handling missing values, categorical encoding, outlier removal.
	•	Feature engineering: domain-inspired interactions, clustering, PCA.
	•	Modeling: gradient boosting (XGBoost), log-target regression, RMSLE optimization.
	•	Hyperparameter optimization: Optuna search space design & tuning.
	•	Software practices: modular pipeline design, Git branching, experiment tracking.


### Acknowledgements
	•	Kaggle for providing the House Prices dataset and starter notebook.
	•	The Ames Housing dataset originally compiled by Dean De Cock (2009).


**Next Steps:** Experiment with ensembling (XGBoost + LightGBM + Ridge) and SHAP analysis for feature importance.
