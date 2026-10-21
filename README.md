# Project 1 of the course ML :) 
### Predicting coronary heart disease (MICDH) with dataset from Behavioral Risk Factor Surveillance System. (from a phonecall to a diagnostic...)
## The content:
In this repository, you can find:
- **run.ipynb**: This Jupyter Notebook contains the preprocessing, the models fitting, etc. It is made so our results can be reproducible.
- **first_data_analysis.ipynb**: The first glance at the dataset to determine the best way to preprocess it.
- **environment.yml**: Which is the environment used to make this project.
- **implementation.py**: This python script has all the functions implemented for this project.
- **helpers.py**: This file has all the functions provided by the course.
- **.gitignore**: 

## The preprocessing of the data
We preprocess the data by:
- removing the features for which 90% or more of the data were NaN
- removing the useless feature for this prediction, like the phone number 
- one hot encoding 
- removing correlated features to reduce the amount of redundancy in the dataset

## The models implemented 
We implemented a Logistic Regression model trained using Stochastic Gradient Descent (SGD) and evaluated via k-fold cross-validation.

## Reproduce the best submission:
To reproduce the best submission, get the environment from **environment.yml** and then run **run.ipynb**. 

