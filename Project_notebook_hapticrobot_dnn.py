# %%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import matplotlib.pyplot as plt
import glob
import re
import warnings
from datetime import datetime
import os
import csv
import io                   
from contextlib import redirect_stdout
import sys

class Logger(object):
    def __init__(self, log_file_path):
        self.terminal = sys.stdout
        self.log = open(log_file_path, mode='a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __del__(self):
        self.log.close()

# %%
def trim_and_flatten_data(haptic_df, robot_df):
    # Haptic: 4, 5th columns (index 3, 4)
    # Robot: 9, 10, 11th columns (index 8, 9, 10)
    haptic_data = haptic_df.iloc[:, [3, 4]].values
    robot_data = robot_df.iloc[:, [8, 9, 10]].values
    
    # Process NaN values by replacing them with zeros
    haptic_data = np.nan_to_num(haptic_data, nan=0.0)
    robot_data = np.nan_to_num(robot_data, nan=0.0)

    # Trim to the length of the shorter dataset
    min_len = min(len(haptic_data), len(robot_df))
    haptic_data_trimmed = haptic_data[:min_len]
    robot_data_trimmed = robot_data[:min_len]

    # Combine and flatten
    combined_features = np.concatenate((haptic_data_trimmed, robot_data_trimmed), axis=1)
    return combined_features.flatten()
# %%
def load_all_data(data_base_path):
    base_path = Path(data_base_path) / "DATA"

    all_X_data = []
    all_y_data = []

    METHOD_MAP = {'HAPTICS': 'H', 'NOhaptics': 'NH'}
    TASK_MAP = {
        5: ('pp1', 'PAP'), 6: ('pp1', 'PAP'),
        7: ('pp2', 'PAPObstructed'), 8: ('pp2', 'PAPObstructed'),
        9: ('pp3', 'Camera'), 10: ('pp3', 'Camera')
    }

    for p_id in range(1, 27): 
        participant_str = f"Participant_{p_id}"
        results_file = base_path / "Haptic Data" / participant_str / f"{participant_str}_results.csv"
        
        # Find output files
        results_lookup = {}
        try:
            results_df = pd.read_csv(results_file)
            for _, row in results_df.iterrows():
                try:
                    condition = row['Condition']
                    subcondition = row['Subcondition']
                    trial_str = str(row['Trial'])
                    trial = int(re.search(r'^\d+', trial_str).group())
                    output_1 = pd.to_numeric(row['Sensor1 Mean'], errors='coerce')
                    output_2 = pd.to_numeric(row['Sensor2 Mean'], errors='coerce')

                    if pd.isna(output_1) or pd.isna(output_2):
                        continue
                    key = (condition, subcondition, trial)
                    results_lookup[key] = (output_1, output_2)
                except Exception:
                    continue
        except Exception as e:
            print(f"Error in reading ({results_file}): {e}")
            continue
            
        # Find input files
        haptic_files_glob = glob.glob(str(base_path / "Haptic Data" / participant_str / "*.csv"))
        # Find haptic files
        for hfile_path in haptic_files_glob:
            file_name = Path(hfile_path).name
            match = re.match(r'(\d+)_.*?_(HAPTICS|NOhaptics)_(\d+)\.csv', file_name)
            
            if not match:
                continue

            try:
                task_num = int(match.group(1))
                method = match.group(2)
                trial = int(match.group(3))
                
                if task_num not in TASK_MAP:
                    continue

                scenario_num, result_condition = TASK_MAP[task_num]
                method_short = METHOD_MAP[method]

                # Find robot files
                robot_pattern = (
                    f"{base_path}/Robot Data/{participant_str}/"
                    f"*_task_{scenario_num}trial{trial}_method_{method_short}_participant_{p_id}.xlsx"
                )
                robot_files = glob.glob(str(robot_pattern))

                if not robot_files:
                    print(f" No matching robot file for: {robot_pattern}")
                    continue
                output_key = (result_condition, method, trial)
                if output_key not in results_lookup:
                    continue

                haptic_df = pd.read_csv(hfile_path)

                if len(robot_files) > 1:
                    df_list = [pd.read_excel(rf) for rf in robot_files]
                    robot_df= pd.concat(df_list, ignore_index=True)
                else:
                    robot_df = pd.read_excel(robot_files[0])

                flat_input_vector = trim_and_flatten_data(haptic_df, robot_df)

                if flat_input_vector is not None:
                    output_1, output_2 = results_lookup[output_key]
                    output_vector = np.array([output_1, output_2])
                    all_X_data.append(flat_input_vector)
                    all_y_data.append(output_vector)
            
                print(f" Success: participant {p_id} task {task_num}, method {method}, trial {trial}")
            except Exception as e:
                print(f"  Error: {file_name} - {e}")

    max_len = max(len(x) for x in all_X_data)
    
    X_padded = np.array([np.pad(x, (0, max_len - len(x)), 'constant') for x in all_X_data])
    y_array = np.array(all_y_data)

    return X_padded, y_array

def split_data(X, y, test_size=0.2, dev_size=0.125, random_state=42):
    X_train_dev, X_test, y_train_dev, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    X_train, X_dev, y_train, y_dev = train_test_split(
        X_train_dev, y_train_dev, test_size=dev_size, random_state=random_state
    )

    print(f"Data split summary:")
    print(f"  (Train)  : {X_train.shape}, {y_train.shape}")
    print(f"  (Dev)    : {X_dev.shape}, {y_dev.shape}")
    print(f"  (Test) : {X_test.shape}, {y_test.shape}")

    return X_train, X_dev, X_test, y_train, y_dev, y_test

def get_regularizer(config):
    reg_type = config.get('regularizer_type', None)
    l1_val = config.get('l1', 0.01)
    l2_val = config.get('l2', 0.01)

    if reg_type == 'l1': return regularizers.l1(l1_val)
    elif reg_type == 'l2': return regularizers.l2(l2_val)
    elif reg_type == 'l1_l2': return regularizers.l1_l2(l1=l1_val, l2=l2_val)
    else: return None

def build_model(input_shape, model_config):
    model = Sequential(name="Modular_DNN_Model")
    model.add(InputLayer(input_shape=input_shape))

    hidden_layers = model_config.get('hidden_layers', [64, 32])
    activation = model_config.get('activation', 'relu')
    reg_obj = get_regularizer(model_config)

    for units in hidden_layers:
        model.add(Dense(units, activation=activation, kernel_regularizer=reg_obj, kernel_initializer='glorot_uniform'))
        if model_config.get('use_batch_norm', False):
            model.add(BatchNormalization())
        if model_config.get('dropout_rate', 0.0) > 0:
            model.add(Dropout(model_config['dropout_rate']))


    # Regression Output
    model.add(Dense(2, name='output', activation='linear'))
    
    model.summary()
    return model

def compile_and_train_model(model, X_train, y_train, X_val, y_val, train_config):
    loss_function = train_config.get('loss', 'mean_squared_error')
    learning_rate = train_config.get('learning_rate', 0.001)
    optimizer = Adam(learning_rate=learning_rate)
    metrics_list = train_config.get('metrics', ['mean_absolute_error'])

    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics_list)

    callbacks_list = []
    patience = train_config.get('early_stopping_patience', None)
    
    if patience and patience > 0:
        early_stop = EarlyStopping(
            monitor='val_loss', 
            patience=patience,  
            verbose=1,          
            restore_best_weights=True
        )
        callbacks_list.append(early_stop)
    
    history = model.fit(
        X_train,
        y_train,
        epochs=train_config.get('epochs', 50),
        batch_size=train_config.get('batch_size', 32),
        validation_data=(X_val, y_val),
        verbose=1,
        callbacks=callbacks_list 
    )

    return model, history

def evaluate_model(model, X_test, y_test):
    results = model.evaluate(X_test, y_test, verbose=1)
    metric_names = model.metrics_names
    for name, value in zip(metric_names, results):
        print(f"{name}: {value:.4f}")
    return results

def plot_training_history(history,filename=None):
    plt.figure(figsize=(12, 5))

    # Loss
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(f"{filename}_history.png", index=False)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    if filename is not None:
        plt.savefig(f"{filename}_loss.png")

    plt.figure(figsize=(12, 5))
    metric_keys = [k for k in history.history.keys() if k not in ['loss', 'val_loss', 'lr']]
    if metric_keys:
        train_metric = metric_keys[0]
        val_metric = f"val_{train_metric}"
        if val_metric in history.history:
            plt.plot(history.history[train_metric], label=f'Train {train_metric}')
            plt.plot(history.history[val_metric], label=f'Validation {val_metric}')
            plt.title(f'Model Metric ({train_metric})')
            plt.xlabel('Epochs')
            plt.ylabel('Metric')
            plt.legend()
    plt.tight_layout()
    if filename is not None:
        plt.savefig(f"{filename}_metric.png")

def plot_predictions(y_true, y_pred,filename=None):
    plt.figure(figsize=(8, 8)) 

    plt.scatter(y_true[:, 0], y_pred[:, 0], 
                alpha=0.5, label='Output 1 Prediction (blue)')
    
    plt.scatter(y_true[:, 1], y_pred[:, 1], 
                alpha=0.5, label='Output 2 Prediction (green)')

    plt.xlabel("Actual Values (True Values)")
    plt.ylabel("Predicted Values (Predictions)")
    plt.title("Prediction vs Actual (Overlay)")
    
    all_values = np.concatenate([y_true.flatten(), y_pred.flatten()])
    lims = [all_values.min() * 0.95, all_values.max() * 1.05] # 약간의 여백 추가
    
    plt.xlim(lims)
    plt.ylim(lims)
    plt.plot(lims, lims, 'r--', label='Perfect Prediction (y=x)')
    
    plt.legend() 
    plt.grid(True)
    plt.tight_layout()

    if filename is not None:
        plt.savefig(f"{filename}_pred.png")

def capture_model_summary(model):
    # Convert model.summary() to string
    stream = io.StringIO()
    with redirect_stdout(stream):
        model.summary()
    return stream.getvalue()


def log_experiment(log_path, model, history, model_config, train_config, X_train, X_val):
    EXPERIMENT_FIELDNAMES = [
        "timestamp", "architecture", "hidden_layers", "activation", 
        "regularizer", "dropout", "batch_norm", "learning_rate", 
        "batch_size", "n_epochs", "optimizer", "loss_function",
        "train_size", "dev_size", "final_train_loss", "final_dev_loss",
        "best_dev_loss", "best_epoch", "train_rmse","dev_rmse"
    ]    
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    train_losses = history.history.get('loss', [0.0])
    val_losses = history.history.get('val_loss', [0.0])
    
    params = {
        "timestamp": timestamp,
        "architecture": capture_model_summary(model), # 모델 구조 요약
        "hidden_layers": str(model_config.get('hidden_layers', [])), # 리스트를 문자열로
        "activation": model_config.get('activation', ''),
        "regularizer": model_config.get('regularizer_type', 'None'),
        "dropout": model_config.get('dropout_rate', 0.0),
        "batch_norm": model_config.get('use_batch_norm', False),
        "learning_rate": train_config.get('learning_rate', 0.0),
        "batch_size": train_config.get('batch_size', 0),
        "n_epochs": train_config.get('epochs', 0),
        "optimizer": model.optimizer.__class__.__name__, # 'Adam' 등
        "loss_function": train_config.get('loss', ''),
        "train_size": len(X_train), # X_train의 크기
        "dev_size": len(X_val),   # X_val의 크기
        "final_train_loss": round(train_losses[-1], 6),
        "final_dev_loss": round(val_losses[-1], 6),
        "best_dev_loss": round(min(val_losses), 6),
        "best_epoch": int(np.argmin(val_losses) + 1), # (numpy as np 필요)
        "train_rmse": round(min(train_metric),6),
        "dev_rmse": round(min(val_metric),6),
    }
    
    file_exists = os.path.exists(log_path)
    with open(log_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, 
                              fieldnames=EXPERIMENT_FIELDNAMES, 
                              restval='N/A', 
                              extrasaction='ignore')
        
        # header
        if not file_exists:
            writer.writeheader() 
            
        writer.writerow(params) 

def save_final_model(save_dir, model, history, train_config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_epochs = train_config.get('epochs', 0)
    
    val_losses = history.history.get('val_loss', [0.0])
    final_dev_loss = val_losses[-1] 
    
    final_name = f"final_DNN_{timestamp}_epoch{n_epochs}_loss{final_dev_loss:.4f}.weights.h5"
    final_path = os.path.join(save_dir, final_name)
    
    model.save_weights(final_path)


def main():
    warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

    DATA_BASE_PATH_ROOT = r"G:/내 드라이브/CS230 Project"
    PROCESSED_DATA_FILE = "processed_haptic_data.npy"

    # Import data
    if os.path.exists(PROCESSED_DATA_FILE):
        data = np.load(PROCESSED_DATA_FILE, allow_pickle=True).item()
        X = data['X']
        y = data['y']
                
    else:
        X, y = load_all_data(DATA_BASE_PATH_ROOT)
        data_to_save = {'X': X, 'y': y}
        np.save(PROCESSED_DATA_FILE, data_to_save, allow_pickle=True)

    # Split data
    X_train_raw, X_val_raw, X_test_raw, y_train_raw, y_val_raw, y_test_raw = split_data(X, y_raw)

    # Scale data
    x_scaler = StandardScaler()
    X_train_2d = X_train_raw.reshape(-1, X_train_raw.shape[-1]) # (N_train * T, F)
    x_scaler.fit(X_train_2d) 

    X_train = x_scaler.transform(X_train_raw.reshape(-1, 5)).reshape(X_train_raw.shape)
    X_val = x_scaler.transform(X_val_raw.reshape(-1, 5)).reshape(X_val_raw.shape)
    X_test = x_scaler.transform(X_test_raw.reshape(-1, 5)).reshape(X_test_raw.shape)

    y_scaler = MinMaxScaler()
    y_scaler.fit(y_train_raw) 

    y_train = y_scaler.transform(y_train_raw)
    y_val = y_scaler.transform(y_val_raw)
    y_test = y_scaler.transform(y_test_raw)

    # Build model
    input_shape = X_train.shape[1:]

    hl=[128,64]
    for nlayers in range(3,30):
        print(f"\n--- 은닉층 개수: {nlayers} ---")
        hl.append(32)
        for learning_rate in (0.005,0.002,0.001):
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            print(f"\n=== 실험 시작: {timestamp} | hidden_layers={hl} | learning_rate={learning_rate} ===")

            MODEL_CONFIG = {
                'hidden_layers': hl,
                'activation': 'relu',
                'regularizer_type': 'l2',
                'l2': 0.001,
                'use_batch_norm': True,
                'dropout_rate': 0.
            }

            TRAIN_CONFIG = {
                'learning_rate': learning_rate,
                'epochs': 10000, 
                'batch_size': 32,
                'loss': 'mean_squared_error',
                'metrics': ['mean_absolute_error', tf.keras.metrics.RootMeanSquaredError(name='rmse')],
                'early_stopping_patience': 30
            }
            model = build_model(input_shape, MODEL_CONFIG)

    # Train model
            model, history = compile_and_train_model(
                model, X_train, y_train, X_val, y_val, TRAIN_CONFIG
            )

    # Evaluate model
            evaluate_model(model, X_test, y_test)

            # Result processing and plotting
            plot_filename = os.path.join("results", "plots", f"hr_loss_curve_{timestamp}")

            plot_training_history(history,filename=plot_filename)

            y_pred_val = model.predict(X_val)
    plot_filename = os.path.join("results", "plots", f"hr_prediction_{timestamp}")
            plot_predictions(y_val,y_pred_val,filename=plot_filename)

            y_pred_original = y_scaler.inverse_transform(y_pred_val)
            y_val_original = y_scaler.inverse_transform(y_val)

    plot_filename = os.path.join("results", "plots", f"hr_prediction_original_{timestamp}")
            plot_predictions(y_val_original,y_pred_original,filename=plot_filename)

            LOG_FILE = os.path.join("results", "experiment_logs.csv")
            log_experiment(
                    log_path=LOG_FILE,
                    model=model,
                    history=history,
                    model_config=MODEL_CONFIG,
                    train_config=TRAIN_CONFIG,
                    X_train=X_train,
                    X_val=X_val,
                )

            save_final_model("results/weights", model, history, TRAIN_CONFIG)

if __name__ == "__main__":
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_DIR = f"logs/run_{TIMESTAMP}"
    os.makedirs(LOG_DIR, exist_ok=True)
    
    TEXT_LOG_FILE = f"{LOG_DIR}/log_output.txt"
    sys.stdout = Logger(TEXT_LOG_FILE)
    warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')
    
    main()
    