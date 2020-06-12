# Sentiment-Analysis-Using-Naive-Bayes
This is an academic project done in the course **CSCI-B55 Machine Learning** at Indiana University.

**Tools and Technology used:** Python, NumPy

Implemented **Naïve Bayes classifier** using **stratified k-fold cross validation** for sentiment analysis on Amazon, Yelp and IMDB 
dataset, analyzing the effect of smoothing on accuracy of the model. Calculated **maximum likelihood (MLE)** and **MAP** solutions and 
compared their results by generating learning curves by subsampling the dataset.

The project consists of 4 code files (.py):\
_main.py_, _generatedKfolds.py_, _NBC.py_, _train_test.py_

The 'Sentiment Labelled Sentences' folder contains all the data set required for the project. The code considers that the data is in the same directory as the code.

## HOW TO RUN THE CODE
    1. To run the code you only have to use the main.py file which takes two command line arguments
       -> filename: name of the data set file
       -> integer value (1,2): to decide which experiment to run
	        1 : experiment 1
	        2 : experiment 2
    2. Example: If I want to perform experiment 1 on yelp data set then...
	     python main.py yelp_labelled.txt 1
