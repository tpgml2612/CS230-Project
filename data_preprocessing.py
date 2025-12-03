import os
import pandas as pd
import ast
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(examples_path='valid_examples_listseq.csv', labels_path='valid_labels_listseq.csv'):
    # Load the datasets if available, otherwise generate synthetic ones
    print('--------------------------------------step 1: Load Data--------------------------------------')
    print(f'Loading data: examples={examples_path}, labels={labels_path}')
    if os.path.exists(examples_path) and os.path.exists(labels_path):
        examples = pd.read_csv(examples_path)
        labels = pd.read_csv(labels_path)
    else:
        # Fail fast: require CSV files to be present.
        raise FileNotFoundError(
            f"Required CSV files not found. Please place '{examples_path}' and '{labels_path}' in the working directory, or provide the full path to your dataset files to load_data()"
        )

    print(f'Loaded: examples shape={examples.shape}, labels shape={labels.shape}')
    return examples, labels

def preprocess_data(examples, labels, test_size=0.2, random_state=42, scale_labels=True):
    print('--------------------------------------step 2: Preprocess Data--------------------------------------')
    def _is_list_string(s):
        return isinstance(s, str) and s.strip().startswith('[') and s.strip().endswith(']')

    def _is_missing(value):
        if value is None:
            return True
        if isinstance(value, str) and value.strip() == '':
            return True
        if isinstance(value, (float, np.floating)):
            return np.isnan(value)
        try:
            result = pd.isna(value)
            if isinstance(result, (bool, np.bool_)):
                return bool(result)
        except Exception:
            pass
        return False

    def _parse_sequence(value, col_name):
        """Convert a stringified list / iterable to a 1D numpy array."""
        if _is_missing(value):
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if _is_list_string(stripped):
                try:
                    parsed = ast.literal_eval(stripped)
                except Exception as exc:
                    raise ValueError(f"Unable to parse list string in column '{col_name}': {str(exc)}")
            elif stripped == '':
                return None
            else:
                try:
                    return np.array([float(stripped)], dtype=float)
                except Exception as exc:
                    raise ValueError(f"Value '{value}' in column '{col_name}' cannot be interpreted as float") from exc
        elif isinstance(value, (list, tuple, np.ndarray)):
            parsed = value
        else:
            try:
                return np.array([float(value)], dtype=float)
            except Exception as exc:
                raise ValueError(f"Unsupported value '{value}' in column '{col_name}' for raw sequence parsing") from exc

        arr = np.array(parsed, dtype=float).flatten()
        if arr.size == 0:
            return None
        return arr

    # Build a numeric-only feature dataframe from examples. For any column that is a stringified list
    # expand it into multiple summary features; otherwise keep it.
    processed_cols = []
    processed_data = []
    # print('Checking example columns for stringified sequence data...')
    # print("examples.columns", examples.columns)
    # print(f"examples[col].values: {examples["follower0_seq"][0]}")
    # print(f'examples shape before processing: {examples.shape}')
    for col in examples.columns: # col : follower0_seq, follower1_seq
        col_vals = examples[col].values
        # If the first non-null value looks like a list string, assume the whole column follows the same format
        first_nonnull = None
        for v in col_vals:
            if v is not None and (not (isinstance(v, float) and np.isnan(v))):
                first_nonnull = v
                break
        is_sequence_col = first_nonnull is not None and (
            _is_list_string(first_nonnull)
            or isinstance(first_nonnull, (list, tuple, np.ndarray))
        )

        if is_sequence_col:
            # print(f"Detected sequence data in column '{col}' — flattening raw sequence values")
            seq_arrays = []
            max_len = 0
            for r in col_vals:
                if _is_missing(r):
                    seq_arrays.append(None)
                    continue
                arr = _parse_sequence(r, col)
                seq_arrays.append(arr)
                if arr is not None:
                    max_len = max(max_len, len(arr))

            if max_len == 0:
                raise ValueError(f"Column '{col}' appears to be list-like but contains no numeric values")

            padded = np.full((len(col_vals), max_len), np.nan, dtype=float)
            for row_idx, arr in enumerate(seq_arrays):
                if arr is None:
                    continue
                seq_len = min(len(arr), max_len)
                padded[row_idx, :seq_len] = arr[:seq_len]

            pad_width = max(1, len(str(max_len - 1)))
            processed_cols.extend([f"{col}_t{idx:0{pad_width}d}" for idx in range(max_len)])
            processed_data.append(padded)
        else:
            # Treat as scalar numeric column
            try:
                numeric_col = examples[col].astype(float).values.reshape(-1, 1)
                processed_cols.append(col)
                processed_data.append(numeric_col)
            except Exception:
                # If a column is not a list string and not castable to float, raise for clarity
                raise ValueError(f"Column '{col}' contains non-numeric data not recognized as list-like strings; sample value: {first_nonnull}")

    # combine processed_data horizontally
    examples_processed = np.hstack(processed_data)
    examples = pd.DataFrame(examples_processed, columns=processed_cols)
    feature_global_means = np.nanmean(examples_processed.astype(float), axis=0)
    feature_global_means = np.where(np.isnan(feature_global_means), 0.0, feature_global_means)
    # print(f"Converted example columns to flat numeric features — new shape={examples.shape}")
    # Convert labels to numpy array; if labels are strings (categorical) encode per column
    if labels.dtypes.apply(lambda x: x == 'object').any():
        # For any columns that are categorical, encode each column separately
        label_encoder = LabelEncoder()
        # Note: we only encode if all columns are categorical; otherwise, prefer numeric labels
        if labels.shape[1] == 1:
            labels_encoded = label_encoder.fit_transform(labels.values.ravel())
            labels_encoded = labels_encoded.reshape(-1, 1)
            # print('Label encoding applied to single column label')
        else:
            # Try to coerce each column to numeric if possible; otherwise, throw
            try:
                labels_encoded = labels.astype(float).values
                label_encoder = None
            except Exception:
                raise ValueError('Mixed-type labels with multiple columns are not supported.')
    else:
        label_encoder = None
        labels_encoded = labels.values
    # print("examples shape after processing:", examples.shape)
    # print("examples values after processing:", examples.values)
    # Split the data into training and testing sets BEFORE scaling to avoid data leakage
    X_raw = examples.values.astype(float)
    y_raw = labels_encoded
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y_raw, test_size=test_size, random_state=random_state)

    # Impute padded NaNs using training-set means to keep raw sequences intact without leakage
    train_feature_means = np.nanmean(X_train_raw, axis=0)
    train_feature_means = np.where(np.isnan(train_feature_means), feature_global_means, train_feature_means)
    train_feature_means = np.where(np.isnan(train_feature_means), 0.0, train_feature_means)

    def _impute_missing(arr):
        rows, cols = np.where(np.isnan(arr))
        if len(rows) > 0:
            arr[rows, cols] = train_feature_means[cols]
        return arr

    X_train_raw = _impute_missing(X_train_raw)
    X_test_raw = _impute_missing(X_test_raw)
    
    # print('Filled padded sequence gaps using training-set feature means')
    # print("X_train_raw after imputation:", X_train_raw)
    # print("len(X_train_raw[200]) after imputation:", len(X_train_raw[200]))
    # print("X_train shape after imputation:", X_train_raw.shape)
    # Keep raw versions of the labels (pre-scaler / pre-encoding for numeric raw values)
    y_train_raw = y_train.copy()
    y_test_raw = y_test.copy()

    # Feature scaling: fit only on the training set and apply to both
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw.astype(float))
    X_test = scaler.transform(X_test_raw.astype(float))
    # print('Feature scaling (StandardScaler) applied to training set and propagated to test set')
    # print(f'Split data: X_train={X_train.shape}, X_test={X_test.shape}, y_train={y_train.shape}, y_test={y_test.shape}')
    # print("X_train sample after scaling:", X_train[0])
    # print("X_train sample after scaling:", X_train)
    # Print label statistics on the training set (raw) to detect scale/outliers
    # try:
    #     y_train_arr = np.asarray(y_train_raw)
    #     if y_train_arr.ndim == 1:
    #         print(f"Label stats (train): mean={np.nanmean(y_train_arr):.4f}, std={np.nanstd(y_train_arr):.4f}, min={np.nanmin(y_train_arr):.4f}, max={np.nanmax(y_train_arr):.4f}")
    #     else:
    #         for col_idx in range(y_train_arr.shape[1]):
    #             arr = y_train_arr[:, col_idx]
    #             print(f"Label {col_idx} stats (train): mean={np.nanmean(arr):.4f}, std={np.nanstd(arr):.4f}, min={np.nanmin(arr):.4f}, max={np.nanmax(arr):.4f}")
    # except Exception:
    #     pass

    # Convert to float32 for PyTorch compatibility
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    # Optionally scale labels (useful for continuous targets)
    label_scaler = None
    if scale_labels:
        from sklearn.preprocessing import StandardScaler as LabelScaler
        label_scaler = LabelScaler()
        # fit on the training labels only and apply to both train and test labels
        y_train = label_scaler.fit_transform(y_train.astype(float)).astype('float32')
        y_test = label_scaler.transform(y_test.astype(float)).astype('float32')
        # print('Labels scaled (to mean 0, std 1) using training labels only')
    else:
        y_train = y_train.astype('float32')
        y_test = y_test.astype('float32')
    # print("X_train : ", X_train)
    return X_train, X_test, y_train, y_test, y_train_raw, y_test_raw, label_encoder, scaler, label_scaler

if __name__ == "__main__":
    examples, labels = load_data()
    X_train, X_test, y_train, y_test, y_train_raw, y_test_raw, label_encoder, scaler, label_scaler = preprocess_data(examples, labels)
    print('Preprocessing complete (standalone run)')