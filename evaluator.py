import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import os

def evaluate_and_plot(model, X_test, y_test, model_name, output_dir):
    print(f"\n--- Ewaluacja modelu: {model_name} ---")
    y_pred = model.predict(X_test)
    
    report = classification_report(y_test, y_pred, digits=4, output_dict=True)
    
    # Zapis macierzy konfuzji
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f'Macierz Konfuzji - {model_name}')
    plt.ylabel('Prawdziwa klasa')
    plt.xlabel('Przewidziana klasa')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{model_name}.png'), dpi=300)
    plt.close()
    
    # Zapis krzywej ROC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1] 
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC area = {roc_auc:.4f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([-0.01, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'roc_auc_{model_name}.png'), dpi=300)
        plt.close()
        
    # Zwraca metryki do zapisania w CSV
    return report['macro avg']['precision'], report['macro avg']['recall'], report['macro avg']['f1-score']