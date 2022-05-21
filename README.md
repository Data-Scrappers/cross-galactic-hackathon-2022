# cross-galactic-hackathon-2022
https://new.skillfactory.ru/hackaton


Data: https://drive.google.com/drive/u/2/folders/1H8yi4g8SfOYWdltp3PFRAKEyusuoqPv-

**Attack Detection for SWaT Dataset using ML methods and Recurrent Neural Networks.**

As part of this educational hackathon, our team analyzed the work of various methods for the SWaT dataset. The results of the study are presented below.
In the context of this study, we analyzed data on physical attacks aimed at disrupting the operation of system components for Year 2015 (SWaT.A1&amp;A2\_Dec2015).
The idea was that based on the data provided from sensors and actuators, it is possible to identify anomalies and determine the possibility of an attack using machine learning models and recurrent neural networks.

We have chosen 3 methods for the analysis of anomalies:

1. Logistic regression
2. Decision tree and gradient boosting (XGBoost)
3. Recurrent neural network (LSTM)

Repository structure:

1. LSTM
Folder with LSTM notebook and py files with model and dataloader.
2. Initial_data_uploading + XGboost test.ipynb
Notebook with data preprocessing and classification using **XGboost**
3. hackaton_logistic_regression.ipynb
**MVP** for this project. Contains data preprocessing and classification using **Logistic Regression**.

Below are descriptions of the use of each of the methods and the results obtained.

**Logistic regression**

Logistic regression training was carried out on preprocessed data (using the method of parameter selection using recursive feature elimination)

| **Metrics** | **Score** |
| ------------|---------- |
| **Accuracy**| **0.98**  |
| **Precision** | **0.99** |
| **Recall** | **0.7** |
| **F1-score** | **0.82** |

**XGBoost**

As part of the data preprocessing before training the model, spaces in the column names were eliminated, and duplicates were removed to avoid data leakage into the training sample. The sample was split into train/test using stratify. After the first prediction, we got overfitting, from which it was concluded that a random data split is not the best option for working with this type of data. Thus, we decided to make a timesplit, while the dataset was presented as a time series, which made it possible to avoid overfitting and get better scores (accuracy - 0.93, f1_score - 0.92).

| **Metrics** | **Score** |
| -----------| --------- |
| **Accuracy**| **0.938**  |
| **Precision** | **0.953** |
| **Recall** | **0.894** |
| **F1-score** | **0.923** |


