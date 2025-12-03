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
from data_preprocessing import load_data, preprocess_data
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


def train(model, device, dataloader, optimizer, loss_fn, epoch, print_every=20):
    model.train()
    running_loss = 0.0
    total = 0
    grad_norm_accum = 0.0
    grad_norm_n = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device).float()
        optimizer.zero_grad()
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
        for data, target in dataloader:
            data, target = data.to(device), target.to(device).float()
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
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f'------->Saved loss plot to {save_path}')
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to YAML config file')
    args = parser.parse_args()

    # Load YAML config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    examples_path = cfg['data']['examples']
    labels_path = cfg['data']['labels']
    scale_labels = cfg['data']['scale_labels']

    epochs = cfg['training']['epochs']
    batch_size = cfg['training']['batch_size']
    lr = cfg['training']['lr']
    dropout = cfg['training']['dropout']
    hidden_dims = cfg['training']['hidden_dims']

    save_dir = cfg['paths']['save_dir']
    use_batchnorm = cfg['model']['use_batchnorm']
    verbose = cfg.get('print', {}).get('verbose', False)

    # parser = argparse.ArgumentParser(description='Train a PyTorch MLP on CSV data or synthetic data (for testing)')
    # parser.add_argument('--examples', type=str, default='valid_examples_listseq.csv', help='Path to examples CSV')
    # parser.add_argument('--labels', type=str, default='valid_labels_listseq.csv', help='Path to labels CSV')
    # parser.add_argument('--scale-labels', action='store_true', help='Scale labels (StandardScaler) during preprocessing')
    # parser.add_argument('--epochs', type=int, default=100)
    # parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    # parser.add_argument('--dropout', type=float, default=0.2, help='Dropout probability (0 disables)')
    # # Batchnorm and dropout are part of the simple model by default
    # parser.add_argument('--save-dir', type=str, default='outputs', help='Directory where plots and final model are saved')
    # parser.add_argument('--batch-size', type=int, default=16)
    # parser.add_argument('--hidden-dims', type=str, default='128,64', help='Comma-separated hidden dims for the MLP (e.g. 128,64)')
    # parser.add_argument('--seed', type=int, default=42)
    # parser.add_argument('--print', type=bool, default=False, help='Print detailed data')
    # # Keep optional label scaling for convenience, but by default we don't scale labels
    # # Simplified options: no grad norm logging, no early abort threshold
    # args = parser.parse_args()

    # print('Starting PyTorch DNN training pipeline')


    print(f'Loading data: examples={examples_path}, labels={labels_path}')
    examples, labels = load_data(examples_path, labels_path)
    X_train, X_test, y_train, y_test, y_train_raw, y_test_raw, label_encoder, scaler, label_scaler = preprocess_data(examples, labels, scale_labels=scale_labels)

    # print('Configuring dataset and dataloaders')
    # Convert to tensors
    X_train_t = torch.from_numpy(X_train)
    X_test_t = torch.from_numpy(X_test)
    # Ensure y arrays are 2D (n_samples, n_targets)
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
    batch_size = batch_size
    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds = TensorDataset(X_test_t, y_test_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    if verbose:
        print(f'Datasets: train={len(train_ds)}, test={len(test_ds)}, batch_size={batch_size}')
        print(f'Device: {device}')


    # Build model
    input_dim = X_train.shape[1]
    n_targets = y_train_arr.shape[1]
    # Parse hidden dims string
    try:
        hidden_dims = [int(x) for x in hidden_dims.split(',') if x.strip()]
    except Exception:
        hidden_dims = [128, 64] # default if parsing fails
    model = MLP(input_dim, hidden_dims=hidden_dims, output_dim=n_targets, dropout=dropout, use_batchnorm=use_batchnorm).to(device)
    if verbose:
        print('-------------------Model architecture-------------------')
        print(model)

    # Training configuration
    epochs = epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
    plot_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)

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
        save_p = os.path.join(save_p, f'Training_target_{t}_scatter_{now}.png') if save_dir else None
        plot_results(y_true_denorm_train[:, t], y_pred_denorm_train[:, t], save_path=save_p)
    
    # Load best model state for final evaluation if available
    # if best_model_state is not None:
    #     model.load_state_dict(best_model_state)
    #     print('Loaded best model state for final evaluation')

    # Save loss curve to file
    loss_plot_path = os.path.join(save_dir, '1. loss_graph')
    loss_plot_path = os.path.join(loss_plot_path, f'losses_{now}.png') if save_dir else None
    try:
        plot_losses(train_losses, val_losses, save_path=loss_plot_path)
    except Exception as e:
        print(f'Failed to save loss plot: {e}')
    # Save losses to CSV
    try:
        if save_dir:
            loss_csv_path = os.path.join(save_dir, '5. loss_csv')
            loss_csv_path = os.path.join(loss_csv_path, f'losses_{now}.csv')
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
        for data, _ in test_loader:
            pred = model(data.to(device)).cpu().numpy()
            if pred.ndim == 1:
                print('---------------------------- pred is 1D ----------------------------')
            y_preds.extend(pred.tolist())
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
            save_p = os.path.join(save_p, f'Test_target_{t}_scatter_{now}.png') if save_dir else None
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
        final_model_path = os.path.join(final_model_path, f'model_final_{now}.pth')
        torch.save(model.state_dict(), final_model_path)
        print(f'------->Saved final model state to {final_model_path}')

    if save_dir:
        config_save_path = os.path.join(save_dir, '4. config')
        config_save_path = os.path.join(config_save_path, f'config_used_{now}.yaml')
        with open(config_save_path, 'w') as f:
            yaml.dump(cfg, f)
        print(f'------->Saved config file to {config_save_path}')
if __name__ == '__main__':
    main()
