import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

def evaluate_and_plot(model, X_test, y_test, model_name="Best_TPOT_Model"):
    """
    Oblicza metryki z artykułu naukowego i generuje wykresy: Macierz konfuzji i ROC/AUC.
    """
    print(f"\n--- Ewaluacja modelu: {model_name} ---")
    y_pred = model.predict(X_test)
    
    # 1. Metryki jakości (Gotowe do skopiowania do tabeli LaTeX)
    report = classification_report(y_test, y_pred, digits=4)
    print("Raport klasyfikacji (Accuracy, Precision, Recall, F1-Score):")
    print(report)
    
    # 2. Macierz konfuzji
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f'Macierz Konfuzji - {model_name}')
    plt.ylabel('Prawdziwa klasa')
    plt.xlabel('Przewidziana klasa')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name}.png', dpi=300)
    print(f"Zapisano wykres macierzy konfuzji jako: confusion_matrix_{model_name}.png")
    
    # 3. Wykres ROC i AUC
    # Sprawdzamy, czy model potrafi przewidywać prawdopodobieństwa (wymagane dla ROC)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1] # Prawdopodobieństwo klasy pozytywnej
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([-0.01, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title(f'Receiver Operating Characteristic (ROC) - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'roc_auc_{model_name}.png', dpi=300)
        print(f"Zapisano wykres ROC jako: roc_auc_{model_name}.png")
    else:
        print("Model nie wspiera metody predict_proba. Pomięto generowanie wykresu ROC.")