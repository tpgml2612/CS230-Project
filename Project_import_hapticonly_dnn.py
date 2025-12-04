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
from Project_notebook_hapticonly_dnn import split_data, build_model, plot_predictions

def calculate_metrics(y_true, y_pred):
    #Calculate RMSE, MAE, R2 given true and predicted values
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "R2": r2}

def main(timestamp,params):
    hl,learning_rate = params
    warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

    #Import data
    PROCESSED_DATA_FILE = "../CS230-Project/processed_haptic_data.npy"
    
    # Import data
    data = np.load(PROCESSED_DATA_FILE, allow_pickle=True).item()
    X = data['X']
    y = data['y']

    # Scale data
    x_scaler = StandardScaler()
    X_scaled = x_scaler.fit_transform(X)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_scaled, y,0.15,0.15)    

    input_shape = X_train.shape[1:]

    MODEL_CONFIG = {
        'hidden_layers': hl,
        'activation': 'relu',
        'regularizer_type': 'l2',
        'l2': learning_rate,
        'use_batch_norm': True,
        'dropout_rate': 0.
    }

    #Build model and load weights
    input_shape = X_train.shape[1:]
    model = build_model(input_shape, MODEL_CONFIG)

    search_pattern = f"Results/final_DNN_{timestamp}_*.weights.h5"
    found_files = glob.glob(search_pattern)
    target_file = found_files[0] 
    
    model.load_weights(target_file)

    #Evaluate model
    sets = {
        "Train": (X_train, y_train),
        "Validation": (X_val, y_val),
        "Test": (X_test, y_test)
    }
    results = []


    for set_name, (X_data, y_true) in sets.items():
        if len(X_data) == 0: continue
        
        y_pred = model.predict(X_data, verbose=0)
        
        metrics = calculate_metrics(y_true, y_pred)
        metrics["Dataset"] = set_name # 구분자 추가
        results.append(metrics)

        plot_filename = os.path.join("Plots", f"hr_prediction_original_{timestamp}_{set_name}.png")
        plot_predictions(y_true,y_pred,filename=plot_filename)

    df_results = pd.DataFrame(results)
    df_results = df_results.set_index("Dataset") # Dataset 컬럼을 인덱스로 설정
    
    df_results.to_csv(f"evaluation_{timestamp}.csv")
    
if __name__ == "__main__":
    for (timestamp,params) in [("20251108_015531",([128, 64, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32],0.002))]:
        main(timestamp,params)


