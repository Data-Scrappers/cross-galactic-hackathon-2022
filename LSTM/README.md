**LSTMs** are commonly used for time series regression and classification
tasks.
Therefore, we are also using the simple model
to predict, if a stand is under attack.

### First part is data preprocessing:
1. Delete invalid data, convert timestamps, scale numerical data
2. Remove first five hours (stabilization part)
3. Split to train, test and validation datasets:
    * Train: from 2015-12-28 10:29:14 until 2016-01-02 13:41:11
    * Validation: from 2015-12-30 12:00:00 until 2016-1-1 11:59:59
    * Test: from 2016-1-1 12:00:00 until 2016-01-02 14:59:59


### Model
ShallowRegressionLSTM consists of PyTorch LSTM layer and single
linear layer at output.

### Train
There are two strategies:
1. Train on all features and predict probability of attack (we can then tune threshold
in order to be more sensitive)
2. Train to predict 'LIT301' sensor values based only on actuators states using only 'Normal' data
and compare with actual value. Later this approach can be extrapolated
on all sensors outputs. 'Attack' flag will be raised
if actual measured value is deviated from predicted one for several timestamps.


### Bottleneck
Even one LSTM layer requires a lot of time to train.
Therefore, it was hard to experiment and test different settings.


---

#### Links that might be useful for future improvements:

1. PyTorch multivariate time series anomaly detection
https://github.com/KurochkinAlexey/ConvRNN/blob/master/ConvRNN_SML2010.ipynb

2. Anomaly Detection for SWaT Dataset using Sequence-to-Sequence Neural Networks
https://github.com/jukworks/swat-seq2seq

3. PyTorch Forecasting package:
https://pytorch-forecasting.readthedocs.io/en/stable/index.html

4. Using LSTM Autoencoder model trained on dataset
without anomalies, detect anomalies based on reconstruction loss
https://curiousily.com/posts/time-series-anomaly-detection-using-lstm-autoencoder-with-pytorch-in-python/
