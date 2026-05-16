import sys
import importlib
sys.modules['imp'] = importlib 

import time
import psutil
import os
from tpot import TPOTClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def get_system_metrics():
    process = psutil.Process(os.getpid())
    ram = process.memory_info().rss / (1024 ** 2)
    cpu = psutil.cpu_percent(interval=0.5)
    return ram, cpu

def train_baseline_models(X_train, y_train, X_test, y_test):
    print("\n--- Trenowanie modeli bazowych (Baseline) ---")
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42)
    }
    
    results = []
    for name, model in models.items():
        ram_start, _ = get_system_metrics()
        start_time = time.time()
        
        model.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        ram_end, cpu_usage = get_system_metrics()
        
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"{name} Baseline - Accuracy: {acc:.4f}, Czas: {train_time:.2f}s")
        
        results.append({
            'Model_Type': 'Baseline',
            'Algorithm': name,
            'Accuracy': acc,
            'Train_Time_sec': train_time,
            'RAM_Usage_MB': ram_end - ram_start,
            'CPU_Usage_percent': cpu_usage,
            'Model_Object': model
        })
        
    return results

def run_tpot_optimization(X_train, y_train, X_test, y_test, generations=5, population_size=10):
    print("\n--- Rozpoczęcie optymalizacji genetycznej TPOT ---")
    
    custom_tpot_config = {
        'sklearn.ensemble.RandomForestClassifier': {'n_estimators': [50, 100], 'max_depth': [None, 10]},
        'xgboost.XGBClassifier': {'n_estimators': [50, 100], 'learning_rate': [0.1, 0.5], 'max_depth': [3, 5]}
    } # ew można użyć "TPOT light" lub "TPOT sparse" dla szybszej optymalizacji przy TPOT light zmieniłem eval time max na minute, 
# dla tpot light puszcze mniej klas 
    tpot = TPOTClassifier(
        generations=generations,
        population_size=population_size,
        cv=3, 
        #config_dict="TPOT light", 
        max_eval_time_mins=2, 
        verbosity=2, 
        random_state=42
    )
    
    ram_start, _ = get_system_metrics()
    start_time = time.time()
    
    tpot.fit(X_train, y_train)
    
    end_time = time.time()
    ram_end, cpu_usage = get_system_metrics()
    train_time = end_time - start_time
    
    best_model = tpot.fitted_pipeline_
    acc = best_model.score(X_test, y_test)
    print(f"TPOT Accuracy: {acc:.4f}, Czas: {train_time:.2f}s")
    
    result = {
        'Model_Type': 'TPOT_Optimized',
        'Algorithm': str(best_model.steps[-1][1]).split('(')[0], # Pobiera nazwę modelu (np. XGBClassifier)
        'Accuracy': acc,
        'Train_Time_sec': train_time,
        'RAM_Usage_MB': ram_end - ram_start,
        'CPU_Usage_percent': cpu_usage,
        'Model_Object': best_model
    }
    
    return result