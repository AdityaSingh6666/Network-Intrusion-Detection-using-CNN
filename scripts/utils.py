import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_nsl_kdd(file_path):
    column_names = [
        'duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent',
        'hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root',
        'num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login',
        'count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate',
        'diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate',
        'dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
        'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty'
    ]
    df = pd.read_csv(file_path, names=column_names)
    df = df.drop('difficulty', axis=1)
    return df

def preprocess_data(df):
    # Convert 'label' to binary: normal = 0, attack = 1
    df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
    print(df['label'].value_counts())

    # Separate features and labels
    X = df.drop('label', axis=1)
    y = df['label'].values

    # One-hot encode the correct categorical columns
    categorical_cols = ['protocol_type', 'service', 'flag']
    X = pd.get_dummies(X, columns=categorical_cols)

    # Scale all features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Feature shape after one-hot encoding: {X_scaled.shape}")

    # Pad or truncate to 144 features for 12x12 reshaping
    target_features = 144
    if X_scaled.shape[1] < target_features:
        padding = np.zeros((X_scaled.shape[0], target_features - X_scaled.shape[1]))
        X_scaled = np.hstack((X_scaled, padding))
    elif X_scaled.shape[1] > target_features:
        X_scaled = X_scaled[:, :target_features]

    # Reshape for CNN: (samples, 12, 12, 1)
    X_reshaped = X_scaled.reshape(-1, 12, 12, 1)
    print(f"X reshaped for CNN: {X_reshaped.shape}")

    return X_reshaped, y
