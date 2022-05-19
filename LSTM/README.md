LSTMs are commonly used for time series regression and classification
tasks.
Therefore we are also using the simple model
to predict, if a stand is under attack.

However it is challenging to split dataset into training and testing,
since we have to be time consistant.

Therefore we use propose to take 4 'Normal' days and one day
with 'Attack' events.
Validate on remaining 2 days with 'Attack' events.


Links that might be useful for future improvements:

1. PyTorch multivariate time series anomaly detection
https://github.com/KurochkinAlexey/ConvRNN/blob/master/ConvRNN_SML2010.ipynb

2. Anomaly Detection for SWaT Dataset using Sequence-to-Sequence Neural Networks
https://github.com/jukworks/swat-seq2seq

3. PyTorch Forecasting package:
https://pytorch-forecasting.readthedocs.io/en/stable/index.html

4. Using LSTM Autoencoder model trained on dataset
without anomalies, detect anomalies based on reconstruction loss
https://curiousily.com/posts/time-series-anomaly-detection-using-lstm-autoencoder-with-pytorch-in-python/
