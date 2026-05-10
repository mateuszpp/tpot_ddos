import os
import glob
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def clean_and_prepare_data(df):
    """Czyszczenie danych i przygotowanie cech oraz etykiet."""
    df.columns = df.columns.str.strip()
    
    # Usuwamy kolumny, które mogłyby sprawić, że model "nauczy się na pamięć" IP/Portów
    cols_to_drop = ['Unnamed: 0', 'Flow ID', 'Source IP', 'Source Port', 
                    'Destination IP', 'Destination Port', 'Timestamp', 'SimillarHTTP']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    
    # Konwersja etykiet na problem binarny (0 = BENIGN, 1 = ATAK)
    if 'Label' in df.columns:
        df['Is_Attack'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
        df = df.drop(columns=['Label'])
    else:
        return None, None
    
    # Czyszczenie błędów matematycznych (inf, NaN)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    # Zostawiamy tylko liczby
    df = df.select_dtypes(include=[np.number])
    
    y = df['Is_Attack']
    X = df.drop(columns=['Is_Attack'])
    
    return X, y

# Ustawiamy folder z naszymi przygotowanymi małymi plikami (z poprzedniego kroku)
input_folder = './male_pliki_10k_min500'
csv_files = glob.glob(os.path.join(input_folder, '*.csv'))

print(f"Znaleziono {len(csv_files)} plików do przetestowania.\n")
print("="*60)

# Inicjalizujemy pustą listę, by zebrać wyniki do podsumowania
wyniki_podsumowanie = []

for file_path in csv_files:
    filename = os.path.basename(file_path)
    print(f"--- Analiza ataku z pliku: {filename} ---")
    
    try:
        # Wczytujemy mały plik
        df = pd.read_csv(file_path, low_memory=False)
        X, y = clean_and_prepare_data(df)
        
        if X is None or len(y.unique()) < 2:
            print(" -> Pomijam plik (brak kolumny Label lub plik zawiera tylko jedną klasę danych).\n")
            continue
            
        # 1. Podział danych: 80% trening, 20% test
        # stratify=y zapewnia, że zachowamy równe proporcje ruchu BENIGN w obu zbiorach!
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        
        # 2. Inicjalizacja i trening modelu
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        
        # 3. Testowanie modelu (predykcja)
        y_pred = clf.predict(X_test)
        
        # 4. Wyliczanie i wyświetlanie wyników
        acc = accuracy_score(y_test, y_pred)
        
        print(f" -> Zbiór treningowy: {len(X_train)} wierszy, Zbiór testowy: {len(X_test)} wierszy.")
        print(f" -> Dokładność (Accuracy): {acc * 100:.2f}%")
        
        # Odkomentuj poniższe linie, jeśli chcesz widzieć szczegóły dla każdego pliku
        # print(" -> Macierz pomyłek:")
        # print(confusion_matrix(y_test, y_pred))
        # print(classification_report(y_test, y_pred, zero_division=0))
        
        print("="*60)
        
        # Dodajemy wynik do listy podsumowującej
        wyniki_podsumowanie.append({'Atak/Plik': filename, 'Dokladnosc (%)': round(acc * 100, 2)})
        
    except Exception as e:
        print(f" -> Wystąpił błąd: {e}\n")

# Wyświetlanie tabelki z podsumowaniem na samym końcu
print("\n*** PODSUMOWANIE WYNIKÓW DLA WSZYSTKICH ATAKÓW ***")
df_wyniki = pd.DataFrame(wyniki_podsumowanie)
print(df_wyniki.to_string(index=False))