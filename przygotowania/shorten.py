import pandas as pd
import os
import glob

# Ustawienia
input_folder = '.' 
output_folder = './male_pliki_10k_min500' 
DOCELOWA_LICZBA_WIERSZY = 10000
MIN_BENIGN = 500  # Wymuszone minimum normalnego ruchu

os.makedirs(output_folder, exist_ok=True)
csv_files = glob.glob(os.path.join(input_folder, '*.csv'))

print(f"Znaleziono {len(csv_files)} plików CSV. Rozpoczynam przetwarzanie z wymuszeniem min. {MIN_BENIGN} BENIGN...\n")

for file_path in csv_files:
    filename = os.path.basename(file_path)
    print(f"Przetwarzanie pliku: {filename}...")
    
    try:
        # Wczytujemy duży chunk (500 000 wierszy to bezpieczna wartość dla RAMu)
        df_chunk = pd.read_csv(file_path, nrows=500000, low_memory=False)
        
        # Standaryzacja nazw kolumn
        df_chunk.columns = df_chunk.columns.str.strip()
        
        if 'Label' not in df_chunk.columns:
            print(f" -> Pomijam plik {filename}, brak kolumny 'Label'.")
            continue
            
        # Podział na ruch normalny i złośliwy
        df_benign = df_chunk[df_chunk['Label'] == 'BENIGN']
        df_attack = df_chunk[df_chunk['Label'] != 'BENIGN']
        
        liczba_benign = len(df_benign)
        liczba_attack = len(df_attack)
        
        if len(df_chunk) <= DOCELOWA_LICZBA_WIERSZY:
            # Jeśli w pliku jest w ogóle mniej niż 10k wierszy, bierzemy co jest
            df_sampled = df_chunk
        else:
            # --- NOWA LOGIKA OBLICZANIA PROPORCJI ---
            
            # Krok 1: Obliczamy naturalną proporcję z tego chunka
            proporcja_benign = liczba_benign / len(df_chunk)
            docelowo_benign = int(DOCELOWA_LICZBA_WIERSZY * proporcja_benign)
            
            # Krok 2: Wymuszamy minimum (jeśli naturalnie wyszło mniej)
            if docelowo_benign < MIN_BENIGN:
                docelowo_benign = MIN_BENIGN
                
            # Krok 3: Zabezpieczenie – nie możemy wziąć więcej niż fizycznie istnieje w chunku!
            docelowo_benign = min(docelowo_benign, liczba_benign)
            
            # Krok 4: Resztę do 10 000 wypełniamy atakiem
            docelowo_attack = DOCELOWA_LICZBA_WIERSZY - docelowo_benign
            
            # Zabezpieczenie na ataki (choć zazwyczaj jest ich ogromna nadwyżka)
            docelowo_attack = min(docelowo_attack, liczba_attack)
            
            # Krok 5: Jeśli brakowało nam BENIGN (np. było tylko 100 w całym pliku), 
            # to docelowo_attack dopełni pulę do równych 10 000
            brakuje_do_10k = DOCELOWA_LICZBA_WIERSZY - (docelowo_benign + docelowo_attack)
            if brakuje_do_10k > 0 and liczba_attack > docelowo_attack:
                docelowo_attack += brakuje_do_10k
                
            # --- KONIEC LOGIKI ---

            # Losujemy odpowiednie ilości i łączymy
            sampled_benign = df_benign.sample(n=docelowo_benign, random_state=42) if docelowo_benign > 0 else pd.DataFrame()
            sampled_attack = df_attack.sample(n=docelowo_attack, random_state=42) if docelowo_attack > 0 else pd.DataFrame()
            
            # Tasujemy połączone dane
            df_sampled = pd.concat([sampled_benign, sampled_attack]).sample(frac=1, random_state=42).reset_index(drop=True)
            
        # Zapisujemy wynik
        output_path = os.path.join(output_folder, filename)
        df_sampled.to_csv(output_path, index=False)
        
        wynik_benign = len(df_sampled[df_sampled['Label'] == 'BENIGN'])
        wynik_attack = len(df_sampled[df_sampled['Label'] != 'BENIGN'])
        print(f" -> Sukces! Zapisano {len(df_sampled)} wierszy (Ruch normalny: {wynik_benign}, Atak: {wynik_attack}).")
        
    except Exception as e:
        print(f" -> Wystąpił błąd przy pliku {filename}: {e}")

print(f"\nGotowe! Nowe pliki znajdziesz w folderze: {output_folder}")