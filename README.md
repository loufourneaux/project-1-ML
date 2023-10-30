# CS-433: Machine Learning Fall 2023, Project 1 
- Team Name: Clementines
- Team Member:
    1. Lou, **SCIPER: 311084** (lou.fourneaux@epfl.ch)
    2. Adrien Feillard, **SCIPER: 315921** (adrien.feillard@epfl.ch)
    3. Laissy Aurélien, **SCIPER: 329573** (aurelien.laissy@epfl.ch)

* [Getting started](#getting-started)
    * [Project description](#project-description)
    * [Data](#data)
    * [Report](#report)
* [Reproduce results](#reproduce-results)
    * [Requirements](#Requirements)
    * [Repo Architecture](#repo-architecture)
    * [Instructions to run](#instructions-to-run)
* [Results](#results)

# Getting started
## Project description
The aim of this project of Machine Learning do some early detection and prevention MICHD diseases. The goal is to estimate the likelihood of developing MICHD given a certain clinical and lifestyle situation.
The model is based on a vector of features collecting the health-related data of a person.
We preprocess and train the dataset on 6 different regression models using cross validation methods to tune hyperparameters. and we assess which one gives the best f1 score to give the best prediction.

More details about the project are available in `references/project1_description.pdf`.
## Data
The data that you we used comes from the Behavioral Risk Factor Surveillance System (BRFSS), a system of
health-related telephone surveys that collects state data about U.S. residents regarding their health-related risk
behaviors, chronic health conditions, and use of preventive services. In particular, respondents were classified
as having coronary heart disease (MICHD) if they reported having been told by a provider they had MICHD.
Respondents were also classified as having MICHD if they reported having been told they had a heart attack (i.e.,
myocardial infarction) or angina.
 The dataset is available at https://www.aicrowd.com/challenges/epfl-machine-learning-project-1/dataset_files. To reproduce the results the files should be added to the repo, as described in [Repo Architecture](#repo-architecture). A detailed description of the dataset is available in 'https://www.cdc.gov/brfss/annual_data/annual_2015.html'.

## Report
All the details about the choices that have been made and the methodology used throughout this project are available in `report.pdf`. In this report, the different asumptions, decisions and results made and found are explained.
# Reproduce results
## Requirements
- Python==3.9.13
- Numpy==1.21.5
- Matplotlib

## Repo Architecture
<pre>  
├─── submission.csv: File generated by run.py. Contains predictions of sample from test.csv. 
├─── test.csv: File containing samples to be predicted.
├─── train.csv: File with labeled sample using for training.
├─── project1_description.pdf: Original description of the project provided by EPFL.
├─── 2015_Codebook_Report.pdf: Reference used to understand features of the dataset.
├─── data_processing.py: File containing implementations to process the raw data.
├─── helpers.py: File provided by EPFL containing methods to load the data and create submissions for aircrowd.
├─── model.py: File containing definition of subfonctions used in implementations.py
├─── cross_validation.py: File containing functions to find the best hyperparameters for each optimization methods
├─── implementations.py: File containing basics ML implementations asked in the project description.
├─── README.md: README
├─── report.pdf: Report explaining choices that has been made.
└─── run.py: File that load the dataset, launch the cross validation, trains models with parameters  and generate submissison.csv.

## Instructions to run 
Move to the root folder and execute:

    python run.py

Make sure to have all the requirements and the data folder in the root. 

# Results
The performances of the models is assessed on AirCrowd from `data/submission.csv` generated by `run.py`. The model achieves a global accuracy of "enter our best accuracy" with a F1-score of "best f1 score".
