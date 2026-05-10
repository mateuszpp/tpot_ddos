import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. KONFIGURACJA
# ==========================================
WIELKI_PLIK = 'DrDoS_DNS.csv'  # <--- Wpisz tutaj nazwę swojego wielkiego pliku
PLIK_TRAIN = 'dane_treningowe_tymczasowe.csv'
PLIK_TEST = 'dane_testowe_tymczasowe.csv'
ROZMIAR_PACZKI = 250000  # Ile wierszy ładujemy do RAMu na raz (bardzo bezpieczna wartość)

def clean_and_prepare_data(df):
    """Funkcja czyszcząca"""
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

# ==========================================
# 2. FAZA 1: PODZIAŁ DANYCH BEZPOŚREDNIO NA DYSKU
# ==========================================
if not os.path.exists(PLIK_TRAIN) or not os.path.exists(PLIK_TEST):
    print(f"KROK 1: Czytanie wielkiego pliku '{WIELKI_PLIK}' w paczkach i zrzut na dysk (Train/Test)...")
    
    for nr_paczki, chunk in enumerate(pd.read_csv(WIELKI_PLIK, chunksize=ROZMIAR_PACZKI, low_memory=False)):
        X, y = clean_and_prepare_data(chunk)
        if X is None:
            continue
            
        # Dzielimy paczkę: 80% do nauki, 20% do testów
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Przyklejamy z powrotem etykietę, żeby zapisać całość do pliku
        X_train['Is_Attack'] = y_train
        X_test['Is_Attack'] = y_test
        
        # Dopisywanie do plików na dysku (tryb 'a' - append)
        # Jeśli plik jeszcze nie istnieje (nr_paczki == 0), zapisujemy z nagłówkami
        X_train.to_csv(PLIK_TRAIN, mode='a', header=not os.path.exists(PLIK_TRAIN), index=False)
        X_test.to_csv(PLIK_TEST, mode='a', header=not os.path.exists(PLIK_TEST), index=False)
        
        print(f" -> Przetworzono i podzielono paczkę nr {nr_paczki+1}")

    print("Zakończono podział! Wygenerowano pliki treningowe i testowe na dysku.\n")
else:
    print("Pliki treningowe i testowe już istnieją na dysku. Przechodzę do uczenia...\n")

# ==========================================
# 3. FAZA 2: TRENING W LOCIE (INCREMENTAL LEARNING)
# ==========================================
print("KROK 2: Inicjalizacja modeli i trening Out-of-Core...")

scaler = StandardScaler()
modele = {
    "SGD Classifier (Liniowy)": SGDClassifier(loss='log_loss', random_state=42),
    "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(50,), random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=0, warm_start=True, random_state=42, n_jobs=-1)
}
all_classes = np.array([0, 1])

# Czytamy wygenerowany plik treningowy po kawałku
for nr_paczki, chunk in enumerate(pd.read_csv(PLIK_TRAIN, chunksize=ROZMIAR_PACZKI)):
    y_train = chunk['Is_Attack']
    X_train = chunk.drop(columns=['Is_Attack'])
    
    # Douczanie Skalera i transformacja
    scaler.partial_fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    
    # Douczanie modeli wspierających partial_fit
    modele["SGD Classifier (Liniowy)"].partial_fit(X_train_scaled, y_train, classes=all_classes)
    modele["Neural Network (MLP)"].partial_fit(X_train_scaled, y_train, classes=all_classes)
    
    # Sprytny hack dla Lasu Losowego: dodajemy mu 5 drzew co każdą paczkę i dołączamy nowe dane
    if len(np.unique(y_train)) > 1:
        modele["Random Forest"].n_estimators += 5
        modele["Random Forest"].fit(X_train_scaled, y_train)
        
    print(f" -> Douczono modele na paczce treningowej nr {nr_paczki+1}")

# ==========================================
# 4. FAZA 3: TESTOWANIE W LOCIE
# ==========================================
print("\nKROK 3: Testowanie na wygenerowanym zbiorze testowym...")

# Słowniki na zebranie wszystkich predykcji
y_true_all = []
y_pred_all = {nazwa: [] for nazwa in modele.keys()}

# Czytamy plik testowy po kawałku, modele tylko odgadują (predict) i zapisujemy wyniki
for nr_paczki, chunk in enumerate(pd.read_csv(PLIK_TEST, chunksize=ROZMIAR_PACZKI)):
    y_test = chunk['Is_Attack']
    X_test = chunk.drop(columns=['Is_Attack'])
    X_test_scaled = scaler.transform(X_test)
    
    y_true_all.extend(y_test.values)
    for nazwa_modelu, model in modele.items():
        y_pred_all[nazwa_modelu].extend(model.predict(X_test_scaled))

# ==========================================
# 5. PODSUMOWANIE I WYNIKI
# ==========================================
wyniki = []
for nazwa_modelu in modele.keys():
    y_pred = y_pred_all[nazwa_modelu]
    wyniki.append({
        'Model': nazwa_modelu,
        'Accuracy': round(accuracy_score(y_true_all, y_pred), 4),
        'Precision': round(precision_score(y_true_all, y_pred, zero_division=0), 4),
        'Recall': round(recall_score(y_true_all, y_pred, zero_division=0), 4),
        'F1-Score': round(f1_score(y_true_all, y_pred, zero_division=0), 4)
    })

print("\n" + "="*70)
print(f"*** WYNIKI TESTÓW DLA CAŁEGO PLIKU ({WIELKI_PLIK}) ***")
print("="*70)
print(pd.DataFrame(wyniki).to_string(index=False))

# Posprzątanie plików tymczasowych (opcjonalnie, można to zakomentować)
# os.remove(PLIK_TRAIN)
# os.remove(PLIK_TEST)