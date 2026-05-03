from data_loader import load_and_preprocess_data
from tpot_optimizer import train_baseline_models, run_tpot_optimization
from evaluator import evaluate_and_plot

def main():
    # KROK 1: Ścieżka do Twojego datasetu DDoS
    # Zmień ścieżkę na właściwą dla Twoich danych (np. 'SDN_Dataset.csv')
    dataset_path = '../male_pliki_10k_min500/DrDoS_NTP.csv'  # Przykładowa ścieżka do zbioru danych
    
    # Zastąp nazwę kolumny docelowej ('Label') jeśli u Ciebie nazywa się inaczej.
    # Wartość sample_fraction służy do redukcji bardzo dużych zbiorów, tu bierzemy 10%.
    try:
        X_train, X_test, y_train, y_test = load_and_preprocess_data(
            dataset_path, sample_fraction=1, target_column='Label' 
        )
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku {dataset_path}. Podaj prawidłową ścieżkę do CSV z danymi.")
        return

    # KROK 2: Trening modeli bazowych bez optymalizacji (dla porównania do raportu)
    baseline_results = train_baseline_models(X_train, y_train, X_test, y_test)
    
    # KROK 3: Optymalizacja potoków za pomocą algorytmu genetycznego (TPOT)
    # Zgodnie z artykułem, konfigurujemy 5 generacji algorytmu
    best_pipeline = run_tpot_optimization(
        X_train, y_train, X_test, y_test, 
        generations=5, 
        population_size=10
    )
    
    # KROK 4: Generowanie metryk, macierzy konfuzji i krzywej ROC do artykułu
    evaluate_and_plot(best_pipeline, X_test, y_test, model_name="TPOT_Optimized")

if __name__ == "__main__":
    main()