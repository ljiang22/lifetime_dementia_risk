# Prediction of lifetime dementia risk - Lian Jiang, 2020

## Table of Contents
1. [Introduction](README.md#introduction)
1. [Input dataset](README.md#input-dataset)
1. [Methods](README.md#methods)
1. [System requirements](README.md#system-requirements)
1. [Run instruction](README.md#run-instruction)
1. [Questions?](README.md#questions?)

## Introduction
The goal of this work is to predict the lifetime dementia risk for people. In this project, I used people's demographics, family and healthy history, genetic and behavioral assessment data to build a predictive model using random forest machine learning technique, which not just can help users predict their lifetime dementia risk, but also help them assess the top risk factors for this disease.

## Input dataset
The data I used is OASIS-3, collected by Washington university over 15 years. OASIS-3 is the latest release in the Open Access Series of Imaging Studies (OASIS) that aimed at making neuroimaging datasets freely available to the scientific community. OASIS-3 is a longitudinal neuroimaging, clinical, cognitive, and biomarker dataset for normal aging and Alzheimer’s Disease. (See more detail using the link: https://www.oasis-brains.org/). 

The input files used in this work are different csv files, residing in the top-most `raw data` directory of the repository. 

## Methods
The steps I used to predict the lifetime dementia risk for users are as follows:
1)  Define the problem, including understanding the business/context, formulating a clear problem statement, defining the input and outpupt, set constrait and success metrics, evaluate feasibility.
2)  Data collection, including defining data needs, identifying data sources, choosing collection methods, ensuring data quality, handling legal & ethical consideration, storing & organizing the data, dcoumening the process.
3)  Data preprocessing, including processing the missing and abnormal values, converting the categorial data to numerical numbers, and merge and match the data from different sources, and etc. 
4)  Exploratory Data Analysis (EDA), including studying further on the context, univriate analysis, bivariate analysis, multivariate analysis using correlation heatmap, variance analysis, PCA analysis, target variable analysis.
5)  Feature engineering and selection.
6)  Data augmentation using numerical simulation.
7)  Model building, including data normalization, data splitting, model selection, machine learning architecture design, and hyper-parameters tuning.
8)  Model evaluation and result analysis.
9)  Develop frontend application.
10) Model and application deployment.


## System requirements
* Both Linux and Windows are supported.
* 64-bit Python 3.7 installation.
* Packages required: Pandas, Numpy, Matplotlib, Sklearn, Seaborn, Streamlit, and Pickle.

## Run instruction
1) Run UDS_preprocess_1.py and UDS_preprocess_2.py to preprocess the data;
2) Run data_wrangling.py to merge the data from different tables and do some preliminary analysis on the data;
3) Run exploratory_data_analysis.py to perform the tasks below: build correlation heat map, remove features with low variance, and remove the features that have a low chance to have an effect on the dependent variable.
4) Run predictive_model.py to train the machine learning models and build the predictive model;
5) Run dementia_risk_predictor.py to build the application of predicting the lifetime dementia risk for users.

## Questions?
For any questions, concerns, and comments, please contact Lian Jiang at jiang2015leon@gmail.com
