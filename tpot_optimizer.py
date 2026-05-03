# --- MAGIA DLA PYTHONA 3.12 ---
import sys
import importlib
sys.modules['imp'] = importlib  # Oszukujemy TPOT-a, żeby nie szukał usuniętego modułu 'imp'
# ------------------------------

import time
from tpot import TPOTClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def train_baseline_models(X_train, y_train, X_test, y_test):
    """
    Trenuje surowe, nie zoptymalizowane modele jako punkt odniesienia (Baseline).
    Zrezygnowano z SVM ze względu na wysoki koszt obliczeniowy na zbiorze DDoS.
    """
    print("\n--- Trenowanie modeli bazowych (Baseline) ---")
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42)
    }
    
    baseline_results = {}
    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"{name} Baseline - Accuracy: {acc:.4f}, Czas treningu: {train_time:.2f}s")
        baseline_results[name] = {'model': model, 'accuracy': acc, 'time': train_time}
        
    return baseline_results

def run_tpot_optimization(X_train, y_train, X_test, y_test, generations=3, population_size=10):
    print("\n--- Rozpoczęcie optymalizacji genetycznej TPOT ---")
    
    # Czyste, profesjonalne podejście: nasza własna konfiguracja bez SVM
    custom_tpot_config = {
        'sklearn.ensemble.RandomForestClassifier': {
            'n_estimators': [100, 200],
            'criterion': ["gini", "entropy"],
            'max_depth': [None, 10, 20]
        },
        'xgboost.XGBClassifier': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1, 0.5],
            'max_depth': [3, 5, 7]
        }
    }

    # TPOT 0.12.1 przyjmie to bez mrugnięcia okiem
    tpot = TPOTClassifier(
        generations=generations,
        population_size=population_size,
        cv=5, 
        config_dict=custom_tpot_config, # Używamy legalnego słownika ograniczającego modele!
        max_eval_time_mins=2, 
        verbosity=2, # Pasek postępu powraca!
        random_state=42
    )
    
    start_time = time.time()
    tpot.fit(X_train, y_train)
    end_time = time.time()
    
    print("\n--- Zakończono proces optymalizacji TPOT ---")
    print(f"Całkowity czas procesu genetycznego: {(end_time - start_time):.2f} sekund")
    
    # Bezpieczne wyciągnięcie gotowego modelu ze Scikit-Learn
    best_model = tpot.fitted_pipeline_
    score = best_model.score(X_test, y_test)
    print(f"Test Accuracy zoptymalizowanego potoku: {score:.4f}")
    
    tpot.export('tpot_best_pipeline.py')
    print("Wyeksportowano strukturę najlepszego potoku do pliku 'tpot_best_pipeline.py'.")
    
    return best_model