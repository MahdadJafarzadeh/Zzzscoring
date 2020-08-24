# Zzzscoring: The interactive version of "ssccoorriinngg" 

![8b7b06ee-e0ba-4065-bcc5-e1f93b8bc592_200x200](https://user-images.githubusercontent.com/48684369/82961288-2b298c80-9fbd-11ea-8f84-76ad2f3cc750.png)

This is a package for automatic sleep scoring. The package is based on [ssccoorriinngg](https://https://github.com/MahdadJafarzadeh/ssccoorriinngg) and in principle provides the same functionality but as a graphical user interface (GUI). The package is able to use many of state-of-art algorithms to train and test and automatic sleep scoring model. 

Please Note: The current version is alpha and still under development.
### Interface
![Capture](https://user-images.githubusercontent.com/48684369/83524226-986c7e80-a4e3-11ea-8464-847602366c0a.JPG)

### Available models
* Artificial neural network (Multi-layer perceptron)
* Support vector machine (SVM)
* Random forest
* ADA Boost
* Gradient boosting
* XGBoost

### How to account for temporal info between stages? 
Zzzscoring has the ability to make use of the so-called "Many-to-one" classification approach. So, to label a specific sleep epoch the features of not only the cutrrent, but also the preceding (backward / online) or preceding together with proceeding (bidirectional / offline) epochs will be used.

### Results
The results can be shown in terms of:
* Metrics, namely: Accuracy, Recall, Precision, F1-score (overall and per class)
* Plotting confusion matrix
* Plotting comparative hypnogram
* Plotting comparative spectrogram (to do)
* Power comparative spectral density (to do)

### Functional Version 
We recommend to make use of the version 1.7 as the latest stable version.
