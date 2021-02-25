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

Each row represents a single point (restaurant review) and each column its features (presence or not of words used in reviews). Each column is an individual feature apart from the last column which contains the labels, whether a review is positive (1) or negative (0).

## Setup
Download the .py files, datasets, and different feature files (class priors,positive_feature_likelihoods and negative_feature_likelihood).
Place all files in

## Run Code

Add main method: 

$    if __name__ == '__main__':
         fit(X, y, attributes)
         save_parameters(attributes)
         numpy.seterr(divide='ignore')

$        count_pos = numpy.array(pandas.read_csv("positive_feature_likelihoods.tsv", header=None, delimiter="\t"))
         count_neg = numpy.array(pandas.read_csv("negative_feature_likelihoods.tsv", header=None, delimiter="\t"))


 $       label_pred = prediction(xval, attributes['Class prior'], 1 - attributes['Class prior'], count_pos, count_neg)
