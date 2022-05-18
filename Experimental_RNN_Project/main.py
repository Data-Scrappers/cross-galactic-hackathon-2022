import gc
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from model_wrapper import ModelWrapper

RANDOM_SEED = 42


def read_csv_file(csv_path: str):
    df = pd.read_csv(csv_path)
    # Remove unknown features
    df.drop(['Unnamed: 0'], axis=1, inplace=True)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    # df['Timestamp'] = df['Timestamp'].apply(lambda x: x.value)
    df.sort_values(by=['Timestamp'], inplace=True)
    df.drop(['Timestamp'], axis=1, inplace=True)
    return df


def create_dataset(df):
    sequences = df.astype(np.float32).to_numpy().tolist()
    dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
    n_seq, seq_len, number_features = torch.stack(dataset).shape

    return dataset, seq_len, n_features


if __name__ == '__main__':
    # Cuda maintenance
    gc.collect()
    torch.cuda.empty_cache()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Torch device: ", device)

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    full_df = read_csv_file('data/data.csv')
    target_name = 'Normal/Attack'
    first_anomaly_idx = full_df[full_df[target_name] == 1].index[0]  # 498554
    normal_df = full_df.iloc[:first_anomaly_idx - 1].drop(labels=target_name, axis=1)
    features = normal_df.columns.values.tolist()

    train_df, val_df = train_test_split(
        normal_df,
        test_size=0.2,
        shuffle=False
    )

    val_df, test_df = train_test_split(
        val_df,
        test_size=0.33,
        shuffle=False
    )

    n_features = len(features)
    time_series_len = 50
    train_dataset = create_dataset(train_df, time_series_len)
    val_dataset = create_dataset(val_df, time_series_len)
    test_normal_dataset = create_dataset(test_df, time_series_len)

    model_wrapper = ModelWrapper(time_series_len, n_features, device)
    model, history = model_wrapper.train(train_dataset, val_dataset)


