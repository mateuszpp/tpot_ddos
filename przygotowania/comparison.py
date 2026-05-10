import os
import glob
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Ignorujemy ostrzeżenia o niezbieżności sieci neuronowej dla czystości konsoli
import warnings
warnings.filterwarnings("ignore")

def clean_and_prepare_data(df):
    """Czyszczenie danych (pozostaje bez zmian)"""
    df.columns = df.columns.str.strip()
    
    cols_to_drop = ['Unnamed: 0', 'Flow ID', 'Source IP', 'Source Port', 
                    'Destination IP', 'Destination Port', 'Timestamp', 'SimillarHTTP']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    
    if 'Label' in df.columns:
        df['Is_Attack'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
        df = df.drop(columns=['Label'])
    else:
        return None, None
    
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    df = df.select_dtypes(include=[np.number])
    
    y = df['Is_Attack']
    X = df.drop(columns=['Is_Attack'])
    return X, y

input_folder = './male_pliki_10k_min500'
csv_files = glob.glob(os.path.join(input_folder, '*.csv'))

print(f"Znaleziono {len(csv_files)} plików do przetestowania.\n")

# Lista na zebranie wszystkich wyników
wyniki_podsumowanie = []

# Definiujemy 3 modele do porównania
modele = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
    # Używamy prostszej sieci (1 warstwa 50 neuronów), by szybko się trenowała na CPU
    "Neural Network": MLPClassifier(hidden_layer_sizes=(50,), max_iter=200, random_state=42) 
}

for file_path in csv_files:
    filename = os.path.basename(file_path)
    print(f"\n{'='*60}\n--- Analiza pliku: {filename} ---")
    
    try:
        df = pd.read_csv(file_path, low_memory=False)
        X, y = clean_and_prepare_data(df)
        
        if X is None or len(y.unique()) < 2:
            print(" -> Pomijam plik (brak etykiet lub tylko jedna klasa).")
            continue
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # NIEZBĘDNE DLA SIECI NEURONOWYCH: Skalowanie danych
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        for nazwa_modelu, model in modele.items():
            print(f" -> Trenowanie: {nazwa_modelu}...")
            
            # Sieć neuronowa potrzebuje danych przeskalowanych, drzewa mogą korzystać z dowolnych, 
            # ale podanie im przeskalowanych nie zaszkodzi
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            # Wyliczanie 4 metryk oceny
            # zero_division=0 zapobiega błędom, jeśli model niczego nie zgadnie
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Zapisywanie do podsumowania
            wyniki_podsumowanie.append({
                'Plik (Atak)': filename,
                'Model': nazwa_modelu,
                'Accuracy': round(acc, 4),
                'Precision': round(prec, 4),
                'Recall': round(rec, 4),
                'F1-Score': round(f1, 4)
            })
            
    except Exception as e:
        print(f" -> Wystąpił błąd: {e}")

# Wyświetlanie przepięknej tabeli na końcu
print("\n" + "="*80)
print("*** ZBIORCZE PODSUMOWANIE WYNIKÓW (GOTOWE DO ARTYKUŁU) ***")
print("="*80)
df_wyniki = pd.DataFrame(wyniki_podsumowanie)

# Grupujemy po pliku i modelu, żeby ładnie to wyglądało
df_wyswietl = df_wyniki.set_index(['Plik (Atak)', 'Model'])
print(df_wyswietl.to_string())