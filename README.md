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

| **Metrics** | **Score** |
| -----------| --------- |


**LSTM**

LSTMs are commonly used for time series regression and classification tasks.
Therefore we are also using the simple model to predict, if a stand is under attack.

However it is challenging to split dataset into training and testing,
since we have to be time consistant.

Therefore we use propose to take 4 'Normal' days and one day
with 'Attack' events.
Validate on remaining 2 days with 'Attack' events.

After training will be finished, we will compare results with our MVP (Logistic regression).

