import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import matplotlib.pyplot as plt
import glob # 파일 검색을 위해 추가
import re # 정규표현식을 위해 추가
import warnings # 경고 메시지 관리를 위해 추가

# --- 1. 데이터 로딩 및 처리 모듈 (검증 완료) ---

def trim_and_flatten_data(haptic_df, robot_df):
    """
    [검증 완료]
    - Haptic/Robot에서 올바른 열을 추출합니다.
    - NaN 값을 0으로 대체(impute)합니다.
    """
    try:
        # Haptic: 4번째, 5번째 열 (index 3, 4)
        haptic_data = haptic_df.iloc[:, [3, 4]].values
        # Robot: 9, 10, 11번째 열 (index 8, 9, 10)
        robot_data = robot_df.iloc[:, [8, 9, 10]].values
        
        # [NaN 해결 코드]
        haptic_data = np.nan_to_num(haptic_data, nan=0.0)
        robot_data = np.nan_to_num(robot_data, nan=0.0)
        
    except IndexError as e:
        print(f"  [오류] 데이터 열(column) 인덱싱 실패: {e}")
        raise
    except Exception as e:
        print(f"  [오류] 알 수 없는 데이터 변환 오류: {e}")
        raise

    min_len = min(len(haptic_data), len(robot_df))
    
    if min_len == 0:
        print("  [경고] 데이터 길이가 0입니다. 건너뜁니다.")
        return None
        
    haptic_data_trimmed = haptic_data[:min_len]
    robot_data_trimmed = robot_data[:min_len]
    
    combined_features = np.concatenate((haptic_data_trimmed, robot_data_trimmed), axis=1)
    
    return combined_features.flatten()


def load_all_data(data_base_path):
    """
    [검증 완료]
    - 'Participant_1'과 'Participant_01' 형식을 모두 처리합니다.
    - results.csv의 실제 열 이름을 사용하고 'object' 타입을 숫자로 변환합니다.
    - Haptic 파일을 기준으로 Robot 파일과 Output을 매칭합니다.
    """
    base_path = Path(data_base_path) / "DATA"
    
    all_X_data = []
    all_y_data = []
    
    METHOD_MAP = {'HAPTICS': 'H', 'NOhaptics': 'NH'}
    TASK_MAP = {
        5: ('pp1', 'PAP'), 6: ('pp1', 'PAP'),
        7: ('pp2', 'papObstructed'), 8: ('pp2', 'papObstructed'),
        9: ('pp3', 'camera'), 10: ('pp3', 'camera')
    }

    print(f"데이터 로딩 시작... (경로: {base_path})")
    for p_id in range(1, 27): # 1~26
        
        participant_str = f"Participant_{p_id}" 
        results_file = base_path / "Haptic Data" / participant_str / f"{participant_str}_results.csv"
        
        if not results_file.exists():
            participant_str_02d = f"Participant_{p_id:02d}"
            results_file = base_path / "Haptic Data" / participant_str_02d / f"{participant_str_02d}_results.csv"
            
            if not results_file.exists():
                print(f"경고: {participant_str} 또는 {participant_str_02d}의 results.csv를 찾을 수 없습니다. (P:{p_id})")
                continue
            else:
                participant_str = participant_str_02d

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
            print(f"결과 파일 읽기/처리 오류 ({results_file}): {e}")
            continue
            
        haptic_files_glob = glob.glob(str(base_path / "Haptic Data" / participant_str / "*.csv"))
        
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
                
                robot_pattern = (
                    f"{base_path}/Robot Data/{participant_str}/"
                    f"*_task_{scenario_num}trial{trial}_method_{method_short}_participant_{p_id}.xlsx"
                )
                robot_files = glob.glob(str(robot_pattern))
                
                if not robot_files:
                    continue
                output_key = (result_condition, method, trial)
                if output_key not in results_lookup:
                    continue
                    
                haptic_df = pd.read_csv(hfile_path)
                
                if len(robot_files) > 1:
                    df_list = [pd.read_excel(f) for f in robot_files]
                    robot_df = pd.concat(df_list, ignore_index=True)
                else:
                    robot_df = pd.read_excel(robot_files[0])

                flat_input_vector = trim_and_flatten_data(haptic_df, robot_df)
                
                if flat_input_vector is not None:
                    output_1, output_2 = results_lookup[output_key]
                    output_vector = np.array([output_1, output_2])
                    all_X_data.append(flat_input_vector)
                    all_y_data.append(output_vector)
            except Exception as e:
                print(f"  [오류] 파일 매칭/처리 중: {file_name} - {e}")

    if not all_X_data:
        print("경고: 로드된 데이터가 없습니다. `load_all_data` 함수와 파일 경로/패턴을 확인하세요.")
        return np.array([]), np.array([])

    print(f"데이터 로딩 완료. 총 {len(all_X_data)}개의 샘플을 찾았습니다.")
    
    max_len = max(len(x) for x in all_X_data)
    print(f"모든 입력 벡터를 최대 길이 {max_len}에 맞춰 패딩합니다.")
    X_padded = np.array([np.pad(x, (0, max_len - len(x)), 'constant') for x in all_X_data])
    y_array = np.array(all_y_data)
    
    return X_padded, y_array

# --- 2. 데이터셋 분리 모듈 ---

def split_data(X, y, test_size=0.2, val_size=0.125, random_state=42):
    """
    데이터를 훈련, 검증, 테스트 세트로 분리합니다. (기본 70:10:20)
    """
    if X.shape[0] == 0:
        print("오류: 분리할 데이터가 없습니다.")
        return (np.array([]),)*6 

    # [참고] 현재 126개 샘플은 매우 적습니다.
    # test_size=0.2 (25개), val_size=0.125 (13개), train (88개)
    # 모델이 불안정할 수 있으나, 우선 백본 코드를 위해 그대로 진행합니다.
    
    # 1차 분리: 훈련+검증 (80%) / 테스트 (20%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # 2차 분리: 훈련 (70%) / 검증 (10%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=random_state
    )
    
    print(f"데이터 분리 완료:")
    print(f"  훈련 (Train)  : {X_train.shape}, {y_train.shape}")
    print(f"  검증 (Val)    : {X_val.shape}, {y_val.shape}")
    print(f"  테스트 (Test) : {X_test.shape}, {y_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# --- 3. 모델 구성 모듈 ---

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
        model.add(Dense(units, activation=activation, kernel_regularizer=reg_obj))
        if model_config.get('use_batch_norm', False):
            model.add(BatchNormalization())
        if model_config.get('dropout_rate', 0.0) > 0:
            model.add(Dropout(model_config['dropout_rate']))
            
    model.add(Dense(2, name='output')) # 2개 값 예측 (회귀)
    
    print("모델 구성 완료:")
    model.summary()
    return model

# --- 4. 훈련 모듈 ---

def compile_and_train_model(model, X_train, y_train, X_val, y_val, train_config):
    loss_function = train_config.get('loss', 'mean_squared_error')
    learning_rate = train_config.get('learning_rate', 0.001)
    optimizer = Adam(learning_rate=learning_rate)
    metrics_list = train_config.get('metrics', ['mean_absolute_error'])
    
    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics_list)
    
    print("\n--- 모델 훈련 시작 ---")
    history = model.fit(
        X_train,
        y_train,
        epochs=train_config.get('epochs', 50),
        batch_size=train_config.get('batch_size', 32),
        validation_data=(X_val, y_val),
        verbose=1
    )
    print("--- 모델 훈련 완료 ---")
    
    return model, history

# --- 5. 평가 및 시각화 모듈 ---

def evaluate_model(model, X_test, y_test):
    if X_test.shape[0] == 0:
        print("경고: 평가할 테스트 데이터가 없습니다.")
        return None

    print("\n--- 최종 모델 평가 (Test Set) ---")
    results = model.evaluate(X_test, y_test, verbose=1)
    metric_names = model.metrics_names
    for name, value in zip(metric_names, results):
        print(f"{name}: {value:.4f}")
    return results

def plot_training_history(history):
    if not history:
        print("시각화할 훈련 기록(history)이 없습니다.")
        return

    plt.figure(figsize=(12, 5))
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Metric
    metric_keys = [k for k in history.history.keys() if k not in ['loss', 'val_loss', 'lr']]
    if metric_keys:
        train_metric = metric_keys[0] 
        val_metric = f"val_{train_metric}"
        if val_metric in history.history:
            plt.subplot(1, 2, 2)
            plt.plot(history.history[train_metric], label=f'Train {train_metric}')
            plt.plot(history.history[val_metric], label=f'Validation {val_metric}')
            plt.title(f'Model Metric ({train_metric})')
            plt.xlabel('Epochs')
            plt.ylabel('Metric')
            plt.legend()
    plt.tight_layout()
    plt.show()

# --- 6. 메인 실행 함수 ---

def main():
    """
    전체 파이프라인을 실행합니다.
    """
    
    # === 설정 (Configuration) ===
    # [검증 완료] 실제 프로젝트 경로
    DATA_BASE_PATH_ROOT = r"G:\내 드라이브\CS230 Project"
    
    # 모델 아키텍처 설정
    MODEL_CONFIG = {
        'hidden_layers': [128, 64, 32], 
        'activation': 'relu',
        'regularizer_type': 'l2',
        'l2': 0.001,
        'use_batch_norm': True,
        'dropout_rate': 0.2
    }
    
    # 훈련 설정
    TRAIN_CONFIG = {
        'learning_rate': 0.001,
        'epochs': 100, # 126개 샘플은 적으니 epoch를 100~200 정도로 늘려도 좋습니다.
        'batch_size': 16, # 샘플이 적으므로 배치 크기를 32보다 작게 (16) 조정
        'loss': 'mean_squared_error',
        'metrics': ['mean_absolute_error', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
    }
    # ==========================
    
    # 1. 데이터 로드 (검증된 함수 사용)
    X, y = load_all_data(DATA_BASE_PATH_ROOT)
    
    if X.shape[0] == 0:
        print("데이터 로딩에 실패했습니다. 프로그램을 종료합니다.")
        return

    # 1-Extra. 데이터 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. 데이터 분리
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_scaled, y)
    
    if X_train.shape[0] == 0:
        print("데이터 분리에 실패했습니다. (샘플 부족) 프로그램을 종료합니다.")
        return

    # 3. 모델 구축
    input_shape = (X_train.shape[1],) 
    model = build_model(input_shape, MODEL_CONFIG)
    
    # 4. 모델 훈련
    model, history = compile_and_train_model(
        model, X_train, y_train, X_val, y_val, TRAIN_CONFIG
    )
    
    # 5. 모델 평가
    evaluate_model(model, X_test, y_test)
    
    # 6. 결과 시각화
    plot_training_history(history)

if __name__ == "__main__":
    # Excel 파일 경고 무시
    warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')
    
    main()