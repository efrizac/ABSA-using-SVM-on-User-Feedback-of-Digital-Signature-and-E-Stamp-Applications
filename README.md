# ABSA using SVM on User Feedback of Digital Signature and E-Stamp Applications
## Description
This repository contains the code, dataset, and documentation for my thesis titled "Aspect-Based Sentiment Analysis Classification using SVM on User Feedback of Digital Signature and E-Stamp Applications". The research focuses on classifying aspects and sentiments in user feedback using a multilabel approach with data balancing techniques such as MLROS, MLSMOTE, and REMEDIAL.
## Key Features
**Topic Modeling**: Utilize Latent Dirichlet Allocation (LDA) to extract topics and interpret widely discussed aspects from user feedback data.

**Multilabel Data Balancing Techniques**: Implements MLROS, MLSMOTE, and REMEDIAL to address data imbalance issues.

**Multilabel Modeling**: Utilizes Support Vector Machine (SVM) with multilabel techniques like Classifier Chains and One-vs-Rest.

**Model Evaluation**: Evaluated using metrics such as Hamming Loss, Precision, Recall, and F1-Score.
# Main Libraries
* numpy
* pandas
* scikit-learn
* skmultilearn
* matplotlib
* gensim
# Research Outcomes
**1. Topic Modeling:**
Latent Dirichlet Allocation (LDA) was employed to identify frequently discussed aspects within the review dataset of digital signature and e-stamp applications. The LDA model successfully grouped the most relevant and significant keywords for each discovered topic, which were then interpreted as key topics. The results identified four main aspects: **Login and Verification, Efficiency, User Service, and Responsiveness**.

**2. Multilabel Classification:**
The Support Vector Machine (SVM) algorithm was implemented as the primary classifier for aspect and sentiment classification. It was combined with various scenario combinations, including:

* Data splits: 80:20 and 70:30
* Resampling methods: MLROS, MLSMOTE, and REMEDIAL
* Parameter C values: 0.1, 1, and 10
* Strategies: Normal SVM and Classifier Chains

The highest model performance was achieved with the combination of a **70:30** data split, the **MLROS** resampling method, a **C parameter of 1**, and the standard SVM strategy, resulting in a Hamming Loss of **0.0559** or an accuracy of 94%.

**3. Aspect-Based Sentiment Analysis Platform:**
Users can utilize the aspect-based sentiment analysis results through a web-based platform. Built with Flask, the platform integrates the best-trained model stored in pickle format within the pipeline. The platform provides the following features:

* Text Input Analysis: Displays predictions for aspects and sentiments.
* File Input Analysis: Outputs sentiment distribution graphs, word clouds for each aspect, and a table of predictions for each review row.
