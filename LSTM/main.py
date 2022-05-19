import numpy as np
import pandas as pd
import torch
from IPython.display import display
from torch import nn
from torch.utils.data import DataLoader

from sequence_dataset import SequenceDataset
from shallow_regression_lstm import ShallowRegressionLSTM


def read_csv_file(csv_path: str, col_list):
    df = pd.read_csv(csv_path, usecols=col_list)
    # Remove unknown features
    # df.drop(['Unnamed: 0'], axis=1, inplace=True)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    # Convert ot nanoseconds
    start_time = df['Timestamp'][0]
    df['Timestamp'] = df['Timestamp'].apply(lambda x: int((x - start_time).total_seconds()))
    df.sort_values(by=['Timestamp'], inplace=True)
    #  df.drop(['Timestamp'], axis=1, inplace=True)
    return df


def to_seconds(hours: float):
    return int(hours * 60 * 60)


def train_model(data_loader, model, loss_function, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    for X, y in data_loader:
        output = model(X)
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    print(f"Train loss: {avg_loss}")


def test_model(data_loader, model, loss_function):
    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            output = model(X)
            total_loss += loss_function(output, y).item()

    avg_loss = total_loss / num_batches
    print(f"Test loss: {avg_loss}")


def predict(data_loader, model):
    output = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for X, _ in data_loader:
            y_star = model(X)
            output = torch.cat((output, y_star), 0)

    return output


if __name__ == '__main__':
    features = ['MV201',
                'P201',
                'P203',
                'P204',
                'P205',
                'P206',
                'DPIT301',
                'FIT301',
                'LIT301',
                'MV301',
                'MV302',
                'MV303',
                'MV304',
                'P301',
                'P302',
                'AIT401',
                'AIT402',
                'FIT401',
                'LIT401',
                'P402',
                'P403',
                'UV401',
                'AIT501',
                'AIT502',
                'AIT503',
                'AIT504',
                'FIT501',
                'FIT502',
                'FIT503',
                'FIT504',
                'P501']

    full_df = read_csv_file('../input_data/data.csv', features + ['Timestamp', 'Normal/Attack'])

    for c in features:
        full_df[c] = full_df[c].astype(np.float64)

    display(full_df)

    batch_size = 4
    sequence_length = 30

    torch.manual_seed(99)

    test_start = 0.15 * len(full_df)
    df_train = full_df.loc[:test_start].copy()
    df_test = full_df.loc[test_start:].copy()

    train_dataset = SequenceDataset(
        df_train,
        target='Normal/Attack',
        features=features,
        sequence_length=sequence_length
    )

    test_dataset = SequenceDataset(
        df_test,
        target='Normal/Attack',
        features=features,
        sequence_length=sequence_length
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    learning_rate = 5e-5
    num_hidden_units = 16

    shallow_model = ShallowRegressionLSTM(num_sensors=len(features), hidden_units=num_hidden_units)
    loss_function_mse = nn.MSELoss()
    optimizer_adam = torch.optim.Adam(shallow_model.parameters(), lr=learning_rate)

    print("Untrained test\n--------")
    test_model(test_loader, shallow_model, loss_function_mse)
    print()

    for ix_epoch in range(3):
        print(f"Epoch {ix_epoch}\n---------")
        train_model(train_loader, shallow_model, loss_function_mse, optimizer=optimizer_adam)
        test_model(test_loader, shallow_model, loss_function_mse)
        print()

    torch.save(shallow_model.state_dict(), 'shallow_model.pt')

    train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    ystar_col = "Model forecast"
    df_train[ystar_col] = predict(train_eval_loader, shallow_model).numpy()
    df_test[ystar_col] = predict(test_loader, shallow_model).numpy()

    df_out = pd.concat((df_train, df_test))[['Normal/Attack', ystar_col]]

    df_out.to_csv('flowers_submission.csv')

    print(df_out)
