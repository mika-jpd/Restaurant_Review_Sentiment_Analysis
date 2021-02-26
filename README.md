# Naive Bayes Sentiment Analysis

## Table of Contens
* [Description](#description)
* [Dataset](#dataset)
* [Setup](#setup)
* [Run code](#run-code)

## Description
Bernouilli Naive Bayes Machine Learning algorithm to predict whether a review is positive or negative. It correctly predicts the correct label with an accuracy of 84%.

## Technologies
Project is created with:
* Python version: 3.9.1
* NumPy library version : 1.20.0
* Pandas library version : 1.2.2

## Dataset
The dataset has been made such that each feature is a categorical feature (0, 1) representing the presence or absence of words used in restaurant reviews. Common words such as "the", "a" etc... are not categorized.

Each row represents a single point (restaurant review) and each column an individual feature. Since all features are binary features, each point's columns will hold either a 1 (presence of the word represented by the ith feature) or 0 (absence of the word represented by the ith feature).

## Setup
Download the .py files, datasets, and different feature files (class priors,positive_feature_likelihoods and negative_feature_likelihood).
Place all files in single folder or project folder.

## Run Code

Add outside the class: 
```
x = bernouilli_naive_bayes("train_dataset.tsv", "validation_dataset.tsv")
x.fit()
x.save_parameters()

prior_data = numpy.array(pandas.read_csv("class_priors.tsv", header=None, delimiter="\t"))
prior_prob_pos = prior_data[0][0]
prior_prob_neg = prior_data[1][0]

count_pos = numpy.array(pandas.read_csv("positive_feature_likelihoods.tsv", header=None, delimiter="\t"))
count_neg = numpy.array(pandas.read_csv("negative_feature_likelihoods.tsv", header=None, delimiter="\t"))

ypred = x.prediction(x.xval, prior_prob_pos, prior_prob_neg, count_pos, count_neg)
print(x.accuracy(ypred))
 ```
Enjoy!
