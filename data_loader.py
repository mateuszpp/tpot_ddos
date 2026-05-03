import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(file_path, sample_fraction=0.1, target_column='Label', random_state=42):
    """
    Wczytuje zbiór DDoS, czyści nazwy kolumn, usuwa nienumeryczne identyfikatory
    i wykonuje stratyfikowane próbkowanie.
    """
    print(f"Rozpoczęto ładowanie danych z pliku: {file_path}...")
    
    # low_memory=False rozwiązuje problem z DtypeWarning
    df = pd.read_csv(file_path, low_memory=False)
    
    # 1. NAPRAWA BŁĘDU KeyError: Usuwamy białe znaki z nazw kolumn (np. ' Label ' -> 'Label')
    df.columns = df.columns.str.strip()
    
    # Sprawdzenie, czy kolumna docelowa istnieje po wyczyszczeniu
    if target_column not in df.columns:
        raise ValueError(f"Nie znaleziono kolumny '{target_column}'. Dostępne kolumny to: {df.columns.tolist()}")

    # 2. Oddzielenie etykiet (y) i kodowanie z tekstu na liczby (0, 1, ...)
    y_raw = df[target_column]
    le = LabelEncoder()
    # Przekształcamy tekst na liczby, zachowując oryginalny indeks dla spójności danych
    y = pd.Series(le.fit_transform(y_raw), index=y_raw.index)
    
    # Wyświetlamy w konsoli, jaka klasa otrzymała jaki numer (np. BENIGN -> 0, DrDoS_DNS -> 1)
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"Zmapowane klasy (Label Encoding): {mapping}")
    
    # 3. Usuwanie kolumn z tekstem (identyfikatory, daty) oraz niepotrzebnego indeksu
    columns_to_drop = [
        target_column, 
        'Unnamed: 0', 
        'Flow ID', 
        'Source IP', 
        'Destination IP', 
        'Timestamp', 
        'SimillarHTTP'
    ]
    
    # Usuwamy tylko te kolumny, które faktycznie istnieją w DataFrame
    cols_to_drop_existing = [col for col in columns_to_drop if col in df.columns]
    X = df.drop(columns=cols_to_drop_existing)
    
    # 4. Zabezpieczenie: wymuszamy pozostawienie tylko wartości numerycznych
    X = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32'])
    
    # 5. Naprawa danych: zbiory CIC często zawierają wartości Infinity (nieskończoność).
    # Modele ML wyrzucą błąd, więc zamieniamy je na NaN, a potem usuwamy wiersze z NaN.
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Synchronizacja X i y po usunięciu wadliwych wierszy
    valid_indices = X.dropna().index
    X = X.loc[valid_indices]
    y = y.loc[valid_indices]
    
    print(f"Dane po oczyszczeniu z tekstów i NaN: {X.shape[0]} wierszy, {X.shape[1]} cech.")

    # 6. Stratyfikowane próbkowanie - kluczowe dla zbalansowania przy dużych zbiorach
    if sample_fraction < 1.0:
        X_sample, _, y_sample, _ = train_test_split(
            X, y, 
            train_size=sample_fraction, 
            stratify=y, 
            random_state=random_state
        )
        print(f"Zredukowano zbiór danych do {sample_fraction*100}% ({len(X_sample)} próbek), zachowując proporcje klas.")
    else:
        X_sample, y_sample = X, y

    # 7. Podział na zbiór treningowy i testowy (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample, 
        test_size=0.2, 
        stratify=y_sample, 
        random_state=random_state
    )
    
    print(f"Zbiór treningowy: X={X_train.shape}, Zbiór testowy: X={X_test.shape}")
    return X_train, X_test, y_train, y_test