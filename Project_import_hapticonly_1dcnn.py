# %%
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import pickle
import glob
import warnings
from datetime import datetime
import os
import csv
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from Project_notebook_hapticonly_1dcnn import forward_fill_nan, pad_with_last_value, load_all_data, split_data, build_model, compile_and_train_model, evaluate_model, plot_training_history, plot_predictions, capture_model_summary, log_experiment, save_final_model

def calculate_metrics(y_true, y_pred):
    #Calculate RMSE, MAE, R2 given true and predicted values
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "R2": r2}

def main(timestamp,params):
    filter_size,dilation_rate,dropout,kernel_size,dense_layers,pooling_size = params
    warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

    #Import data
    PROCESSED_DATA_FILE = "../CS230-Project/processed_haptic_data_cnn_sherlock.npy"
    
    MODEL_CONFIG = { 'cnn_layers': filter_size,
                    'cnn_dilation_rates': dilation_rate,
                    'activation': 'relu',
                    'regularizer_type': 'l2',
                    'l2': 0.001,
                    'use_batch_norm': True,
                    'dropout_rate': dropout,
                    'cnn_filters': 64,
                    'cnn_kernel_size': kernel_size,
                    'dense_units': 64,
                    'dense_layers': dense_layers,
                    'pooling_size': pooling_size }
    
    data = np.load(PROCESSED_DATA_FILE, allow_pickle=True).item()
    X = data['X']
    y_raw = data['y']

    X_train_raw, X_val_raw, X_test_raw, y_train_raw, y_val_raw, y_test_raw = split_data(X, y_raw)

    #Build model and load weights
    input_shape = X_train_raw.shape[1:]
    model = build_model(input_shape, MODEL_CONFIG)

    search_pattern = f"results/weights/final_1DCNN_{timestamp}_*.weights.h5"
    found_files = glob.glob(search_pattern)
    target_file = found_files[0] 
    
    model.load_weights(target_file)

    #Scale data
    x_scaler_path = os.path.join("results", "plots", f"x_scaler_2025-12-03_12-42-31.pkl")
    y_scaler_path = os.path.join("results", "plots", f"y_scaler_2025-12-03_12-42-31.pkl")

    with open(x_scaler_path, 'rb') as f:
        x_scaler = pickle.load(f)
    with open(y_scaler_path, 'rb') as f:
        y_scaler = pickle.load(f)

    X_train = x_scaler.transform(X_train_raw.reshape(-1, 2)).reshape(X_train_raw.shape)
    X_val = x_scaler.transform(X_val_raw.reshape(-1, 2)).reshape(X_val_raw.shape)
    X_test = x_scaler.transform(X_test_raw.reshape(-1, 2)).reshape(X_test_raw.shape)

    y_train = y_scaler.transform(y_train_raw)
    y_val = y_scaler.transform(y_val_raw)
    y_test = y_scaler.transform(y_test_raw)

    #Evaluate model
    sets = {
        "Train": (X_train, y_train),
        "Validation": (X_val, y_val),
        "Test": (X_test, y_test)
    }
    results = []

    for set_name, (X_data, y_scaled_true) in sets.items():
        if len(X_data) == 0: continue
        
        y_pred_scaled = model.predict(X_data, verbose=0)
        
        y_true_orig = y_scaler.inverse_transform(y_scaled_true)
        y_pred_orig = y_scaler.inverse_transform(y_pred_scaled)
        
        metrics = calculate_metrics(y_true_orig, y_pred_orig)
        metrics["Dataset"] = set_name # 구분자 추가
        results.append(metrics)

        plot_filename = os.path.join("results", "plots", f"hr_prediction_original_{timestamp}_{set_name}.png")
        plot_predictions(y_true_orig,y_pred_orig,filename=plot_filename)

    df_results = pd.DataFrame(results)
    df_results = df_results.set_index("Dataset") # Dataset 컬럼을 인덱스로 설정
    
    df_results.to_csv(f"evaluation_{timestamp}.csv")
    
if __name__ == "__main__":
    for (timestamp,params) in [("20251203_124232",([32, 32, 64, 64, 64],[1,2,4,4,4],0,10,[64],4))]:
        main(timestamp,params)


