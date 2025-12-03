import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import torch
import datetime
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from data_preprocessing import load_data, preprocess_data_MLP, preprocess_data_1D_CNN
from data_preprocessing_LSTM import preprocess_data_LSTM
from models.lstm_model import LSTM
import yaml

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], output_dim=1, dropout=0.2, use_batchnorm=True):
        super(MLP, self).__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            last_dim = h
        # Final output: output_dim for multi-target regression
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        return out

class CNN1D(nn.Module):
    def __init__(
        self,
        input_length,
        output_dim,
        channels,
        kernel_sizes,
        pools,
        pool_kernel=2,
        dropout=0.3,
        target_len=32
    ):
        super().__init__()


        keras_dilations = [1, 1, 1, 1]


        dilations = (keras_dilations * 10)[:len(channels)]


        layers = []
        in_ch = 1
        length = input_length

        for out_ch, k, pool_flag, dil in zip(channels, kernel_sizes, pools, dilations):

            layers.append(nn.Conv1d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=k,
                # dilation=dil,
                padding=(k // 2) * dil   # SAME padding
            ))

            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU())

            if pool_flag == 1:
                layers.append(nn.MaxPool1d(pool_kernel))
                length //= pool_kernel

            in_ch = out_ch

        self.features = nn.Sequential(*layers)


        self.global_pool = nn.AdaptiveMaxPool1d(1)


        DENSE1 = 64
        DENSE2 = 32

        self.classifier = nn.Sequential(
            nn.Linear(in_ch, DENSE1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(DENSE1, DENSE2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(DENSE2, output_dim)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.features(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)

        x = self.classifier(x)
        return x



def _unpack_batch(batch):
    if len(batch) == 3:
        data, lengths, target = batch
    else:
        data, target = batch
        lengths = None
    return data, lengths, target


def train(model, device, dataloader, optimizer, loss_fn, epoch):
    model.train()
    running_loss = 0.0
    total = 0
    for batch in dataloader:
        data, lengths, target = _unpack_batch(batch)
        data, target = data.to(device), target.to(device).float()
        optimizer.zero_grad()
        if lengths is not None:
            outputs = model(data, lengths)
        else:
            outputs = model(data)
        loss = loss_fn(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        total += data.size(0)

    avg_loss = running_loss / total if total > 0 else 0.0
    return avg_loss


def evaluate(model, device, dataloader, loss_fn=None):
    model.eval()
    preds = []
    trues = []
    running_loss = 0.0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            data, lengths, target = _unpack_batch(batch)
            data, target = data.to(device), target.to(device).float()
            if lengths is not None:
                outputs = model(data, lengths)
            else:
                outputs = model(data)
            preds.extend(outputs.cpu().numpy().tolist())
            trues.extend(target.cpu().numpy().tolist())
            if loss_fn is not None:
                loss = loss_fn(outputs, target)
                running_loss += loss.item() * data.size(0)
                total += data.size(0)
    avg_loss = running_loss / total if total > 0 else None
    return avg_loss, np.array(trues), np.array(preds)


# EarlyStopping removed for simplified pipeline per user request


def plot_results(y_test, y_pred, save_path=None, title='True Values vs Predicted Values'):
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(title)
    # Use diagonal (y=x) based on the actual min/max of the combined values
    combined_min = min(np.min(y_test), np.min(y_pred))
    combined_max = max(np.max(y_test), np.max(y_pred))
    plt.plot([combined_min, combined_max], [combined_min, combined_max], 'r--')  # Diagonal line
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f'------->Saved plot to {save_path}')
        plt.close()
    else:
        plt.show()


def plot_losses(train_losses, val_losses, save_path=None):
    plt.figure(figsize=(8, 5))
    epochs = list(range(1, len(train_losses) + 1))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Losses')
    plt.ylim(bottom=0, top=2)
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f'------->Saved loss plot to {save_path}')
        plt.close()
    else:
        plt.show()

def create_model(model_type, cfg, input_length, output_dim):
    """
    model_type : string ("mlp", "cnn1d", ...)
    cfg        : cfg["model"] dictionary from YAML
    input_length : computed input dim (usually X_train.shape[1] or shape[2])
    """

    if model_type == "mlp":
        return MLP(
            input_dim=input_length,
            hidden_dims=cfg.get("hidden_dims"),
            output_dim=output_dim,
            dropout=cfg.get("dropout"),
            use_batchnorm=cfg.get("use_batchnorm", False)
        )

    elif model_type == "cnn1d":
        return CNN1D(
            input_length=input_length,
            output_dim=output_dim,
            channels=cfg["channels"],
            kernel_sizes=cfg["kernel_sizes"],
            pools=cfg["pools"],
            pool_kernel=cfg.get("pool_kernel"),
            dropout=cfg.get("dropout")
        )
    elif model_type == "lstm":
        hidden_size = cfg.get("hidden_size", cfg.get("hidden_dim", 128))
        return LSTM(
            input_size=input_length,
            hidden_size=hidden_size,
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.2),
            output_size=output_dim
        )


    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to YAML config file')
    args = parser.parse_args()
    config_input = args.config

    config_path = os.path.join("configs", config_input)
    # Load YAML config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    
    # Load from YAML config
    # Data
    data_cfg = cfg.get('data', {})
    examples_path = data_cfg.get('examples')
    labels_path = data_cfg.get('labels')
    scale_labels = data_cfg.get('scale_labels', False)

    # Training hyperparams
    train_cfg = cfg.get('training', {})
    epochs = train_cfg.get('epochs', 100)
    batch_size = train_cfg.get('batch_size', 32)
    lr = train_cfg.get('lr', 1e-3)
    seed = train_cfg.get('seed', 42)

    # Model hyperparams (fully general)
    model_cfg = cfg.get('model', {})
    model_type = model_cfg.get('type', 'mlp')

    # Values might not exist depending on model
    hidden_dims = model_cfg.get('hidden_dims', None)     # MLP용
    dropout = model_cfg.get('dropout', 0.0)
    use_batchnorm = model_cfg.get('use_batchnorm', False)

    # CNN용 파라미터들도 안전하게 불러오기
    num_channels = model_cfg.get('num_channels', None)
    kernel_size = model_cfg.get('kernel_size', None)
    pool_kernel = model_cfg.get('pool_kernel', None)

    # Save directory
    paths_cfg = cfg.get('paths', {})
    save_dir = paths_cfg.get('save_dir', 'outputs')

    # Print options
    verbose = cfg.get('print', {}).get('verbose', False)

    ###############################

    preprocessing_cfg = cfg.get('preprocessing', {})
    window_size = data_cfg.get('window_size', preprocessing_cfg.get('window_size', 32))
    step_size = data_cfg.get('step_size', preprocessing_cfg.get('step_size', 1))
    val_split = train_cfg.get('val_split', 0.2)
    shuffle_flag = train_cfg.get('shuffle', True)
    num_workers = train_cfg.get('num_workers', 0)

    if model_type == "lstm":
        train_loader, test_loader, input_dim = preprocess_data_LSTM(
            examples_path=examples_path,
            labels_path=labels_path,
            window_size=window_size,
            step_size=step_size,
            batch_size=batch_size,
            val_split=val_split,
            seed=seed,
            shuffle=shuffle_flag,
            num_workers=num_workers,
        )
        train_ds = train_loader.dataset
        test_ds = test_loader.dataset
        y_train_arr = train_ds.labels.numpy()
        y_test_arr = test_ds.labels.numpy()
        y_train_raw = y_train_arr.copy()
        y_test_raw = y_test_arr.copy()
        label_encoder = None
        scaler = None
        label_scaler = None
    else:
        print(f'Loading data: examples={examples_path}, labels={labels_path}')
        examples, labels = load_data(examples_path, labels_path)
        if model_type == "mlp":
            X_train, X_test, y_train, y_test, y_train_raw, y_test_raw, label_encoder, scaler, label_scaler = preprocess_data_MLP(examples, labels, scale_labels=scale_labels)
        elif model_type == "cnn1d":
            X_train, X_test, y_train, y_test, y_train_raw, y_test_raw, label_encoder, scaler, label_scaler = preprocess_data_1D_CNN(examples, labels, scale_labels=scale_labels)
        else:
            raise ValueError(f"Unsupported model type for preprocessing: {model_type}")

        X_train_t = torch.from_numpy(X_train)
        X_test_t = torch.from_numpy(X_test)
        y_train_arr = np.asarray(y_train)
        y_test_arr = np.asarray(y_test)
        if y_train_arr.ndim == 1:
            print("---------------------------- y_train_arr is 1D, need to reshape it to 2D ----------------------------")

        y_train_t = torch.from_numpy(y_train_arr)
        y_test_t = torch.from_numpy(y_test_arr)
        if verbose:
            print(f'X_train shape: {X_train.shape}', f'X_test shape: {X_test.shape}')
            print(f'mean of X_train: {np.mean(X_train):.4f}, std of X_train: {np.std(X_train):.4f}')
            print(f'mean of X_test: {np.mean(X_test):.4f}, std of X_test: {np.std(X_test):.4f}')
            print('-----------------------------------------------------------')
            print(f'y_train shape: {y_train.shape}',f' y_test_t shape: {y_test.shape}')
            print(f'mean of y_train: {np.mean(y_train):.4f}, std of y_train: {np.std(y_train):.4f}')
            print(f'mean of y_test: {np.mean(y_test):.4f}, std of y_test: {np.std(y_test):.4f}')
            print('-----------------------------------------------------------')
            print(f'y_train_raw shape: {y_train_raw.shape}', f'y_test_raw shape: {y_test_raw.shape}')
            print(f'mean of y_train_raw: {np.mean(y_train_raw):.4f}, std of y_train_raw: {np.std(y_train_raw):.4f}')
            print(f'mean of y_test_raw: {np.mean(y_test_raw):.4f}, std of y_test_raw: {np.std(y_test_raw):.4f}')
            print('-----------------------------------------------------------')
        train_ds = TensorDataset(X_train_t, y_train_t)
        test_ds = TensorDataset(X_test_t, y_test_t)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_flag)
        test_loader = DataLoader(test_ds, batch_size=batch_size)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    if verbose:
        print(f'Datasets: train={len(train_ds)}, test={len(test_ds)}, batch_size={batch_size}')
        print(f'Device: {device}')


    # Build model
    if model_type == "cnn1d":
        input_dim = X_train.shape[2]      # CNN: (N, 1, L) -> L
    elif model_type == "mlp":
        input_dim = X_train.shape[1]      # MLP: (N, F) -> F

    n_targets = y_train_arr.shape[1]
    # Parse hidden dims string
    try:
        hidden_dims = [int(x) for x in hidden_dims.split(',') if x.strip()]
    except Exception:
        hidden_dims = [128, 64] # default if parsing fails

    model_cfg = cfg["model"]
    model_type = model_cfg["type"]
    # print("num_channels =", num_channels)
    # print("input_length =", input_dim)
    # print("kernel_size =", kernel_size)
    # print("pool_kernel =", pool_kernel)
    model = create_model(
        model_type=model_type,
        cfg=model_cfg,
        input_length=input_dim,
        output_dim =n_targets
    ).to(device)

    # model = MLP(input_dim, hidden_dims=hidden_dims, output_dim=n_targets, dropout=dropout, use_batchnorm=use_batchnorm).to(device)
    if verbose:
        print('-------------------Model architecture-------------------')
        print(model)

    # Training configuration
    epochs = epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=1e-4)
    # Use Mean Squared Error for regression-style training
    loss_fn = nn.MSELoss()

    print('-------------------Beginning training loop-------------------')
    train_losses = []
    val_losses = []
    os.makedirs(save_dir, exist_ok=True)
    for epoch in range(1, epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, loss_fn, epoch)
        val_loss, _, _ = evaluate(model, device, test_loader, loss_fn)
        train_losses.append(train_loss)
        val_losses.append(val_loss if val_loss is not None else float('nan'))
        # Print only epoch and loss per user request
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}')

    print('-------------------Training finished — evaluating on training set-------------------')

    now = datetime.datetime.now().strftime('%m%d-%H%M%S')

    # 1. Create non-shuffled loader for plotting
    plot_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=train_loader.collate_fn
    )

    # 2. Get correctly ordered true/pred
    train_loss, y_true_train, y_pred_train = evaluate(model, device, plot_loader, loss_fn)

    # print(f'label_scaler: {label_scaler}')
    print(f'1. Final train MSE loss (scaled label space): {train_loss:.6f}')

    # 3. Because evaluate already returns ordered true/pred:
    y_true_scaled_train = y_true_train.copy()
    y_pred_scaled_train = y_pred_train.copy()


    # Denormalize (inverse-transform) scaled labels and predictions into original units if a label_scaler exists
    if scale_labels and label_scaler is not None:
        print('------->Inverse transforming training predictions and true labels to original scale using label_scaler')
        # label_scaler expects shape (n_samples, n_targets)
        y_true_denorm_train = label_scaler.inverse_transform(y_true_scaled_train.reshape(-1, y_true_scaled_train.shape[1]))
        y_pred_denorm_train = label_scaler.inverse_transform(y_pred_scaled_train.reshape(-1, y_pred_scaled_train.shape[1]))
    else:
        print('------->No label_scaler provided or scaling disabled; using raw training labels as original scale')
        y_true_denorm_train = y_train_raw
        y_pred_denorm_train = y_pred_scaled_train
    
    for t in range(y_pred_denorm_train.shape[1]):
        # print(f'Plotting Target {t} vs predicted values (denormalized) on training set')
        save_p = os.path.join(save_dir, '2. train_pred_true_graph')
        save_p = os.path.join(save_p, f'{model_type}_Training_target_{t}_scatter_{now}.png') if save_dir else None
        plot_results(y_true_denorm_train[:, t], y_pred_denorm_train[:, t], save_path=save_p)
    
    # Load best model state for final evaluation if available
    # if best_model_state is not None:
    #     model.load_state_dict(best_model_state)
    #     print('Loaded best model state for final evaluation')

    # Save loss curve to file
    loss_plot_path = os.path.join(save_dir, '1. loss_graph')
    loss_plot_path = os.path.join(loss_plot_path, f'{model_type}_losses_{now}.png') if save_dir else None
    try:
        plot_losses(train_losses, val_losses, save_path=loss_plot_path)
    except Exception as e:
        print(f'Failed to save loss plot: {e}')
    # Save losses to CSV
    try:
        if save_dir:
            loss_csv_path = os.path.join(save_dir, '5. loss_csv')
            loss_csv_path = os.path.join(loss_csv_path, f'{model_type}_losses_{now}.csv')
            df_losses = pd.DataFrame({'epoch': list(range(1, len(train_losses) + 1)), 'train_loss': train_losses, 'val_loss': val_losses})
            df_losses.to_csv(loss_csv_path, index=False)
            print(f'------->Saved losses CSV to {loss_csv_path}')
    except Exception as e:
        print(f'Failed to save losses CSV: {e}')

    test_loss, y_true, y_pred = evaluate(model, device, test_loader, loss_fn)
    print(f'2. Final test MSE loss (scaled label space): {test_loss:.6f}')

    y_preds = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            data, lengths, _ = _unpack_batch(batch)
            preds_batch = model(data.to(device), lengths) if lengths is not None else model(data.to(device))
            preds_np = preds_batch.cpu().numpy()
            if preds_np.ndim == 1:
                print('---------------------------- pred is 1D ----------------------------')
            y_preds.extend(preds_np.tolist())
    # Convert to original label scale if label_scaler is provided; y_true and model preds are in that same space
    # y_true and y_pred are in the scaled training space (if label_scaler was used during preprocessing).
    y_true_scaled = y_true.copy()
    y_pred_scaled = np.array(y_preds).copy()

    # # Denormalize (inverse-transform) scaled labels and predictions into original units if a label_scaler exists
    if scale_labels and label_scaler is not None:
        print('------->Inverse transforming predictions and true labels to original scale using label_scaler')
        # label_scaler expects shape (n_samples, n_targets)
        y_true_denorm = label_scaler.inverse_transform(y_true_scaled.reshape(-1, y_true_scaled.shape[1]))
        y_pred_denorm = label_scaler.inverse_transform(y_pred_scaled.reshape(-1, y_pred_scaled.shape[1]))
    else:
        print('------->No label_scaler provided or scaling disabled; using raw test labels as original scale')
        y_true_denorm = y_test_raw
        y_pred_denorm = y_pred_scaled

        # If no scaler used, `y_test_raw` contains original units; ensure it's 2D
    # Print a few sample comparisons between scaled and denormalized values for verification
    # if args.print:
    #     print('Sample verification: scaled -> denormalized (first 5 entries)')
    # for i in range(min(5, len(y_true_scaled))):
    #     if y_true_scaled.ndim == 1:
    #         print(f'  Sample {i}: scaled_true={y_true_scaled[i]:.6f}, scaled_pred={y_pred_scaled[i]:.6f} -> denorm_true={y_true_denorm[i]:.6f}, denorm_pred={y_pred_denorm[i]:.6f}')
    #     else:
    #         scaled_true_str = ', '.join(f'{v:.6f}' for v in y_true_scaled[i])
    #         scaled_pred_str = ', '.join(f'{v:.6f}' for v in y_pred_scaled[i])
    #         denorm_true_str = ', '.join(f'{v:.6f}' for v in y_true_denorm[i])
    #         denorm_pred_str = ', '.join(f'{v:.6f}' for v in y_pred_denorm[i])
    #         print(f'  Sample {i}: scaled_true=[{scaled_true_str}], scaled_pred=[{scaled_pred_str}] -> denorm_true=[{denorm_true_str}], denorm_pred=[{denorm_pred_str}]')
    # If multiple targets, create separate plots for each target using denormalized values and save them
    
    
    
    if y_pred_denorm.ndim == 1 or y_pred_denorm.shape[1] == 1:
        print('---------------------------- Single target detected for plotting ----------------------------')
    else:
        for t in range(y_pred_denorm.shape[1]):
            # print(f'Plotting Target {t} vs predicted values (denormalized)')
            save_p = os.path.join(save_dir, '3. test_pred_true_graph')
            save_p = os.path.join(save_p, f'{model_type}_Test_target_{t}_scatter_{now}.png') if save_dir else None
            plot_results(y_true_denorm[:, t], y_pred_denorm[:, t], save_path=save_p)

    # # Show a few sample predictions per sample and per target
    # print('Example predictions (first 10 samples):')
    # for i in range(min(10, len(y_true_denorm))):
    #     if y_true_denorm.ndim == 1:
    #         true_str = f'{y_true_denorm[i]:.4f}'
    #         pred_str = f'{y_pred_denorm[i]:.4f}'
    #     else:
    #         true_str = ', '.join(f'{v:.4f}' for v in y_true_denorm[i])
    #         pred_str = ', '.join(f'{v:.4f}' for v in y_pred_denorm[i])
    #     print(f'    Sample {i}: true=[{true_str}], pred=[{pred_str}]')

    # Compute per-target MSE in scaled space
    per_target_mse = []
    per_target_abs_denorm = []
    if y_true_denorm.ndim == 1 or (y_true_denorm.ndim == 2 and y_true_denorm.shape[1] == 1):
        # single target case
        per_target_mse.append(mean_squared_error(y_true_scaled.ravel(), y_pred_scaled.ravel()))
        per_target_abs_denorm.append(np.mean(np.abs(y_true_denorm.ravel() - y_pred_denorm.ravel())))
    else:
        for t in range(y_true_denorm.shape[1]):
            per_target_mse.append(mean_squared_error(y_true_scaled[:, t], y_pred_scaled[:, t]))
            per_target_abs_denorm.append(np.mean(np.abs(y_true_denorm[:, t] - y_pred_denorm[:, t])))
    final_test_mse_scaled = float(np.mean(per_target_mse))
    final_test_abs_denorm = float(np.mean(per_target_abs_denorm))
    print(f'3. Final test MSE per target (scaled): {per_target_mse}')
    print(f'4. Final test MAE per target (denormalized): {per_target_abs_denorm}')

    
    # print(f'3. Final test MSE (scaled, mean across targets): {final_test_mse_scaled:.6f}')
    # Save final model state
    if save_dir:
        final_model_path = os.path.join(save_dir, '6. weights')
        final_model_path = os.path.join(final_model_path, f'{model_type}_model_final_{now}.pth')
        torch.save(model.state_dict(), final_model_path)
        print(f'------->Saved final model state to {final_model_path}')

    if save_dir:
        config_save_path = os.path.join(save_dir, '4. config')
        config_save_path = os.path.join(config_save_path, f'{model_type}_config_used_{now}.yaml')
        with open(config_save_path, 'w') as f:
            yaml.dump(cfg, f)
        print(f'------->Saved config file to {config_save_path}')
if __name__ == '__main__':
    main()
