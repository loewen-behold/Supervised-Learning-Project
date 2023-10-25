# machine_learning_project-supervised-learning

## Project Overview
- Supervised Learning: Use supervised learning techniques to build a machine learning model that can predict whether a patient has diabetes or not, based on certain diagnostic measurements.The project involves three main parts: exploratory data analysis, preprocessing and feature engineering, and training a machine learning model. 

### Goal
The ultimate goal of the project is to gain insights from the data sets and communicate these insights to stakeholders using appropriate visualizations and metrics to make informed decisions based on the business questions asked."

## Process
### 1. EDA:
#### Relationships/Correlations Among Predictors and Between Predictors and Target
When comparing the distributions for the predictors by outcome it was observed that pregnancy, glucose, BMI, and age had the most noticeable differences between outcomes. When specifically looking at the averages of each distribution for these, we saw that: 
- The average age of the individuals in the dataset is ~31 for those without diabetes and ~37 for those with diabetes.
- The average glucose level for those without diabetes is ~110 and those with diabetes is ~141
- The average BMI for individuals without diabetes is ~30 and for those with diabetes is ~35

When looking at the correlations between predictors and target, we observed that:
- The correlation between Outcome and Glucose was high, and therefore likely to be the most important feature in model training.
- BMI, Pregnancies and Age also appeared to be fairly important features for training our model, with moderate correlations to the target.
- Some relatively high correlation between predictors - particularly SkinThickness, BMI, Insulin, and Blood Pressure. Concern for possible multi-collinearity issue, but leveraged these relationships to replace any zero-values found in these columns.

#### Missing Values and Outliers
- Many of the predictors had quite a number of zero-values present, most of which were in the Skin Thickness and Insulin columns.  These were dealt with in the pre-processing stage.  
- There were also a number of outliers present, which were also dealt with in the pre-processing stage.
- Rationalle for dealing with missing values and outliers can be found below and in more detail in the notebook.

### 2. Pre-processing
#### Handling Missing Values
- Eliminated all entries that have 3 or more zero-values listed in either 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin', or 'BMI'. There were 35 of these in total (accounts for nearly 5% of the dataset).
- Replaced zero-values for Glucose, Blood Pressure, and BMI with the means since there are not many of these in comparison to the entire dataset.
- Since there were alot of missing values for Insulin and Skin Thickness, these were dealt with in a slightly different way than the others.  We leveraged the relationships they had with the other predictors in order to replace them with (hopefully) more meaningful values:
- Replaced zero-values for Insulin with the average Insulin of entries with "similar" BMI and Glucose values.
- Replaced zero-values for Skin Thickess with the average Skin Thickness of entries with "similar" BMI and Blood Pressure values.

#### Handling Outliers
- Removed lower outliers for Blood Pressures. Higher outliers left alone.
- Removed upper outliers in skin thickness.
- Removed upper outliers for BMI.
- The remaining outliers were mitigated to some degree in the scaling.

#### Scaling and Normalization / Maintaining Balanced Data
- Used Standard Scaler since it is less sensitive to outliers and lends well to linear models, often helping improve the performance of logistic regression models, which is one of the models planned.

When determining the balance of the data:
- The training and test sets had the same proportionality of outcomes as the original dataset. In all cases, there were about 66% for those without diabetes and 34% with diabetes.
- Concluded that our data was fairly well balanced.

### 3. Model Training
#### Model 1. Logisitic Regression Model with k-fold Cross-Validation and RandomizedSearch
- Started with a Logistic Regression model that utilizes k-fold Cross-Validation and Randomized Search
- Chose F1 as the primary metric to train since we want recall (True Postive Rate) to be maximized, while also having precision maximized as well. Our F1 score is the harmonized mean of the two.

Observations:
Logistic Regression Model had an average F1-score of 0.64, which isn't great.

#### Model 2. Random Forest vs AdaBoost vs SVM vs LDA vs XGBoost vs Gradient Boosting
Tested a number of ensemble models, along with a couple others to compare their performances. Achieved this by using a pipeline of models, looping through all of them, applying a grid search, and determining the best parameters for each model.  From there, we could compare their scores with one another.

Observations:
Overall these models performed about the same, however, the AdaBoost and Random Forest methods seemed to have an edge in Precision and Recall, while AdaBoost, Random Forest, and SVC had an edge in Accuracy. As far as ROC scores, all models were pretty close again, with LDA coming in on top with a score of 0.84.
AdaBoost definitely come out on top amongst the models when considering that recall and precision is what we'd like to maximize in this case.

#### Model 3. Stacking - Voting Classifier
Attempted to stack 5 models, including the Logistic Regression and AdaBoost models in hopes of increasing our earlier performances.  The results were:
F1 Score: 0.65
Accuracy: 0.79
Precision: 0.69
Recall: 0.62
ROC AUC: 0.83

## Conclusion
- There were not many strong correlations between our predictors and our target - glucose was the best of them all. Due to this, our generated models were not performing the way we would have liked them to. In the end, we wanted to see a high recall score from our models since we are dealing with daignosis and thus, want a high True Positive Rate.
- I expected the ensemble models to out-perform the logisitic regression, but a couple only did marginally better. The Logistic Regression model had an F1 score of 0.64, while the next best performer was AdaBoost with an F1 score of almost 0.67. Both had similar ROC curves, with AdaBoost coming out on top with an AUC of 0.83.
- Using stacking was a good thought as well, but I was again surprised to see that this method also didn't yeild much better results than the Logistic Regression on it's own. With stacking, we achieved an F1 score of 0.65 and an AUC of 0.83.
- Overall, Boosting seemed to be the most effective method for achieving higher F1 scores - with AdaBoosting being our top performer. If I had more time, I'd focus in on AdaBoost more and hypertune it to hopefully get better results. I'd also try and use a different scaling and normalization methods - Something that handles outliers a little bit better.
