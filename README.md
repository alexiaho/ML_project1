# Project 1 of the course ML :) 
This is the best version 
### Predicting coronary heart disease (MICDH) with dataset from Behavioral Risk Factor Surveillance System. (from a phonecall to a diagnostic...)
The objective of this project is to develop a machine learning model capable of estimating the likelihood of coronary heart disease (MICHD) based on individual lifestyle and clinical factors. Early identification of individuals at elevated risk is an important component of cardiovascular disease prevention. The dataset used in this study originates from Behavioral Risk factor Surveillance System, a surveillance system that collects health-related information through telephone surveys. Using these data, we implemented a binary classification model based on regularized logistic regression to estimate the probability of an individual belonging to the positive class.
## The content:
In this repository, you can find:
- **run.ipynb**: This Jupyter Notebook contains the preprocessing, the models fitting, etc. It is made so our results can be reproducible.
- **first_data_analysis.ipynb**: The first glance at the dataset to determine the best way to preprocess it.
- **environment.yml**: Which is the environment used to make this project.
- **implementation.py**: This python script has all the functions implemented for this project.
- **helpers.py**: This file has all the functions provided by the course.
- **.gitignore**:
- **grading_tests**: The original folder from the instructions given at the beginning of this project.

## The preprocessing of the data
We preprocess the data by:
- removing the features for which 90% or more of the data were NaN
- removing the useless feature for this prediction, like the phone number 
- one hot encoding 
- removing correlated features to reduce the amount of redundancy in the dataset

## The models implemented 
We implemented a Logistic Regression model trained using Stochastic Gradient Descent (SGD) and evaluated via k-fold cross-validation.

## Reproduce the best submission:
To reproduce the best submission, get the environment from **environment.yml**, download the dataset from the GitHub of the course:https://github.com/epfml/ML_course.git under project1 and then run **run.ipynb**. 

### Workflow

1. Load the data and perform exploratory analysis.
2. Document preprocessing assumptions and fit preprocessing on training partitions only.
3. Run the holdout experiments as exploratory comparisons.
4. Compare the candidate SGD variants using the same stratified five-fold splits and collect out-of-fold probabilities.
5. Select the model variant and decision threshold, refit on all labeled data, and generate `submission.csv`.
6. Summarize baselines, the most impactful ablations, metrics, and limitations in the report.


