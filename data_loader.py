import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def load_and_preprocess_data(file_path, sample_fraction=0.1, random_state=42):
    print(f"Ładowanie danych z: {file_path}...")
    df = pd.read_csv(file_path, low_memory=False)
    df.columns = df.columns.str.strip()
    
    # Krok 1: Wymuszenie binarnej klasyfikacji. 
    # Wszystko co jest 'BENIGN' staje się 0. Wszystko inne (DDoS) staje się 1.
    y_raw = df['Label']
    y = y_raw.apply(lambda x: 0 if x.strip().upper() == 'BENIGN' else 1)
    print(f"Mapowanie klas: 0 -> Ruch normalny (BENIGN), 1 -> Atak (DDoS)")

    columns_to_drop = ['Label', 'Unnamed: 0', 'Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 'SimillarHTTP']
    cols_to_drop_existing = [col for col in columns_to_drop if col in df.columns]
    X = df.drop(columns=cols_to_drop_existing)
    X = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32'])
    X = X.replace([np.inf, -np.inf], np.nan)
    
    valid_indices = X.dropna().index
    X = X.loc[valid_indices]
    y = y.loc[valid_indices]

    if sample_fraction < 1.0:
        X_sample, _, y_sample, _ = train_test_split(X, y, train_size=sample_fraction, stratify=y, random_state=random_state)
    else:
        X_sample, y_sample = X, y

    X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, stratify=y_sample, random_state=random_state)
    return X_train, X_test, y_train, y_test

def create_global_dataset(file_paths, fraction_per_file=0.02):
    """
    Tworzy jeden połączony zbiór z wielu plików (pobiera mały % z każdego, żeby nie zapchać RAMu).
    """
    print("\n--- Tworzenie Globalnego Zbioru (Combined Dataset) ---")
    combined_df = pd.DataFrame()
    for path in file_paths:
        print(f"Pobieranie próbek z {path}...")
        df = pd.read_csv(path, low_memory=False)
        # Pobieramy tylko mały ułamek, żeby globalny zbiór był rozsądnych rozmiarów
        df_sampled = df.sample(frac=fraction_per_file, random_state=42)
        combined_df = pd.concat([combined_df, df_sampled], ignore_index=True)
    
    # Zapisujemy połączony zbiór do tymczasowego pliku, żeby użyć standardowego loadera
    temp_path = "temp_global_dataset.csv"
    combined_df.to_csv(temp_path, index=False)
    
    # Ładujemy używając ułamka 1.0, bo już zmniejszyliśmy dane w pętli wyżej
    X_train, X_test, y_train, y_test = load_and_preprocess_data(temp_path, sample_fraction=1.0)
    os.remove(temp_path) # sprzątamy
    
    return X_train, X_test, y_train, y_test