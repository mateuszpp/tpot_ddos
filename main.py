import os
import csv
from datetime import datetime
from data_loader import load_and_preprocess_data, create_global_dataset
from tpot_optimizer import train_baseline_models, run_tpot_optimization
from evaluator import evaluate_and_plot

def setup_experiment_dir():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"Experiment_Results_{timestamp}"
    os.makedirs(dir_name, exist_ok=True)
    return dir_name

def print_model_params(baseline_results, tpot_model):
    print("\n" + "="*50)
    print(" PORÓWNANIE PARAMETRÓW: BASELINE vs TPOT ")
    print("="*50)
    
    # Wyciąganie Baseline XGBoost (szukamy go w liście wyników)
    xgb_baseline = next((res['Model_Object'] for res in baseline_results if res['Algorithm'] == 'XGBoost'), None)

    
    if xgb_baseline:
        print(f"\n[Baseline XGBoost] Domyślne parametry:")
        params = xgb_baseline.get_params()
        print(f" - max_depth: {params.get('max_depth')}")
        print(f" - learning_rate: {params.get('learning_rate')}")
        print(f" - n_estimators: {params.get('n_estimators')}")

    # Wyciąganie parametrów z TPOT (z ostatniego kroku potoku)
    # tpot_model to zwykle Pipeline, więc bierzemy ostatni element
    tpot_params = tpot_model.steps[-1][1].get_params()
    print(f"\n[TPOT Optimized] Parametry wybrane ewolucyjnie:")
    print(f" - max_depth: {tpot_params.get('max_depth')}")
    print(f" - learning_rate: {tpot_params.get('learning_rate')}")
    print(f" - n_estimators: {tpot_params.get('n_estimators')}")
    print("="*50 + "\n")


def log_to_csv(csv_path, dataset_name, train_size, test_size, result_dict, precision, recall, f1):
    # Sprawdzamy czy plik istnieje, by wiedzieć czy dodać nagłówki
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            # Dodano nowe kolumny: Train_Size i Test_Size
            writer.writerow([
                'Dataset', 'Train_Size', 'Test_Size', 'Model_Type', 'Algorithm', 
                'Accuracy', 'Precision', 'Recall', 'F1_Score', 
                'Train_Time_sec', 'RAM_Usage_MB', 'CPU_Usage_percent'
            ])
        
        writer.writerow([
            dataset_name, 
            train_size,              # Zapisujemy liczbę wierszy treningowych
            test_size,               # Zapisujemy liczbę wierszy testowych
            result_dict['Model_Type'], 
            result_dict['Algorithm'], 
            round(result_dict['Accuracy'], 4),
            round(precision, 4),
            round(recall, 4),
            round(f1, 4),
            round(result_dict['Train_Time_sec'], 2),
            round(result_dict['RAM_Usage_MB'], 2),
            round(result_dict['CPU_Usage_percent'], 2)
        ])

def run_pipeline_for_dataset(dataset_name, X_train, y_train, X_test, y_test, output_dir, csv_path):
    print(f"\n{'='*50}\nROZPOCZĘCIE ANALIZY DLA: {dataset_name}\n{'='*50}")
    
    # Wyciągamy dokładny rozmiar danych
    train_sz = len(X_train)
    test_sz = len(X_test)
    print(f"Dane w tym przebiegu: Trening={train_sz}, Test={test_sz}")
    
    # 1. Modele Bazowe
    baseline_results = train_baseline_models(X_train, y_train, X_test, y_test)
    for res in baseline_results:
        plot_name = f"{dataset_name}_Baseline_{res['Algorithm'].replace(' ', '')}"
        p, r, f1 = evaluate_and_plot(res['Model_Object'], X_test, y_test, plot_name, output_dir)
        # Przekazujemy rozmiary danych do zapisu
        log_to_csv(csv_path, dataset_name, train_sz, test_sz, res, p, r, f1)

    # 2. TPOT (z przyspieszonymi parametrami)
    tpot_res = run_tpot_optimization(X_train, y_train, X_test, y_test, generations=5, population_size=10)
    plot_name = f"{dataset_name}_TPOT_Optimized"
    p, r, f1 = evaluate_and_plot(tpot_res['Model_Object'], X_test, y_test, plot_name, output_dir)
    # Przekazujemy rozmiary danych do zapisu
    log_to_csv(csv_path, dataset_name, train_sz, test_sz, tpot_res, p, r, f1)

    print_model_params(baseline_results, tpot_res['Model_Object'])

def main():
    # Konfiguracja środowiska
    output_dir = setup_experiment_dir()
    csv_path = os.path.join(output_dir, "experiment_report.csv")
    print(f"Wyniki będą zapisywane w katalogu: {output_dir}")

    # Twoja lista plików
    files_to_process = [
        '../DrDoS_NTP.csv',
        '../DrDoS_LDAP.csv',
        '../DrDoS_DNS.csv',
        '../DrDoS_SNMP.csv',
        '../DrDoS_UDP.csv'
    ]

    # --- KONFIGURACJA DYNAMICZNEGO PRÓBKOWANIA ---
    # Możesz dowolnie modyfikować te wartości
    fraction_small_files = 0.1   # Pobierz 10% jeśli plik waży < 1 GB
    fraction_large_files = 0.05  # Pobierz 5% jeśli plik waży >= 1 GB
    GB_IN_BYTES = 1024 * 1024 * 1024 # Przelicznik na Gigabajty
    # ---------------------------------------------

    # CZĘŚĆ 1: Iteracja po dedykowanych modelach dla pojedynczych plików
    for file_path in files_to_process:
        dataset_name = os.path.basename(file_path).split('.')[0]
        
        try:
            # Sprawdzanie wagi pliku na dysku
            file_size_bytes = os.path.getsize(file_path)
            file_size_gb = file_size_bytes / GB_IN_BYTES
            
            # Logika dynamicznego przypisywania sample_fraction
            if file_size_gb >= 1.0:
                current_fraction = fraction_large_files
                print(f"\n[INFO] Plik {dataset_name} ma rozmiar {file_size_gb:.2f} GB (>= 1GB). Ustawiono próbkowanie: {current_fraction*100}%")
            else:
                current_fraction = fraction_small_files
                print(f"\n[INFO] Plik {dataset_name} ma rozmiar {file_size_gb:.2f} GB (< 1GB). Ustawiono próbkowanie: {current_fraction*100}%")

            # Ładowanie danych z dynamicznie dobranym ułamkiem
            X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path, sample_fraction=current_fraction)
            
            # Uruchomienie analizy (Baseline + TPOT)
            run_pipeline_for_dataset(dataset_name, X_train, y_train, X_test, y_test, output_dir, csv_path)
            
        except Exception as e:
            print(f"Błąd podczas przetwarzania pliku {file_path}: {e}")

    # CZĘŚĆ 2: Eksperyment z Globalnym Zbiorem Danych
    try:
        # Pamiętaj, aby dobrać fraction_per_file tak, aby nie zapchać RAM-u przy łączeniu wielu plików!
        X_train_g, X_test_g, y_train_g, y_test_g = create_global_dataset(files_to_process, fraction_per_file=0.03)
        run_pipeline_for_dataset("Global_Combined_Dataset", X_train_g, y_train_g, X_test_g, y_test_g, output_dir, csv_path)
    except Exception as e:
        print(f"Błąd podczas tworzenia globalnego datasetu: {e}")

    print("\nWSZYSTKIE EKSPERYMENTY ZAKOŃCZONE POMYŚLNIE. Sprawdź katalog:", output_dir)

if __name__ == "__main__":
    main()