import numpy as np
import pandas as pd

def calculate_metrics(conf):
    """
    Calcula medidas de desempenho dos classificadores como:
    - Precision
    - Recall
    - Accuracy
    - Specificity
    - F1 Score
    
    Parâmetros:
    conf (numpy array): Matriz de confusão.
    
    Retorna:
    precision (float): Precisão média.
    recall (float): Recall médio.
    accuracy (float): Acurácia média.
    specificity (float): Especificidade média.
    f1score (float): F1 Score médio.
    """
    n = conf.shape[1]  # Obtém o número de classes.

    precision = np.zeros(n)
    recall = np.zeros(n)
    accuracy = np.zeros(n)
    specificity = np.zeros(n)
    f1score = np.zeros(n)

    for indice in range(n):
        tp = conf[indice, indice]  # Verdadeiros positivos para a classe indice.
        fn = conf[indice, np.arange(n) != indice]  # Falsos negativos para a classe indice.
        fp = conf[np.arange(n) != indice, indice]  # Falsos positivos para a classe indice.
        tn = conf[np.ix_(np.arange(n) != indice, np.arange(n) != indice)]  # Verdadeiros negativos para a classe indice.

        # Calcula as métricas para a classe indice.
        precision[indice] = np.sum(tp) / (np.sum(tp) + np.sum(fp))
        recall[indice] = np.sum(tp) / (np.sum(tp) + np.sum(fn))
        accuracy[indice] = (np.sum(tp) + np.sum(tn)) / (np.sum(tp) + np.sum(fp) + np.sum(tn) + np.sum(fn))
        specificity[indice] = np.sum(tn) / (np.sum(tn) + np.sum(fp))
        f1score[indice] = (2 * precision[indice] * recall[indice]) / (precision[indice] + recall[indice])

    # Calcula as médias das métricas para todas as classes.
    precision_avg = np.mean(precision)
    recall_avg = np.mean(recall)
    accuracy_avg = np.mean(accuracy)
    specificity_avg = np.mean(specificity)
    f1score_avg = np.mean(f1score)

    # Cria um dataframe para imprimir as métricas.
    metric_df = pd.DataFrame({
        'Classe': np.arange(1, n + 1),
        'Precision': precision,
        'Recall': recall,
        'Accuracy': accuracy,
        'Specificity': specificity,
        'F1 Score': f1score
    })

    return precision_avg, recall_avg, accuracy_avg, specificity_avg, f1score_avg, metric_df
