import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import pywt

from calcula_descritores import calculate_statistics
from calcula_metricas import calculate_metrics

# Caminho relativo para o arquivo 'mat_vibration.mat'
caminho_arquivo = "C:\\Users\\Mathe\\projetos\\IC-ml-manutencao-preditiva-rolamentos-IFES\\Dados\\mat_classes.mat"

# Carregar o arquivo .mat
data = sio.loadmat(caminho_arquivo)

X = data['classes'][:, :2560]
y = data['classes'][:, 2560]

def plot_wavelet_coeffs(coefficients, sample_index, wavelet_name, level):
    # Recupere os coeficientes da amostra escolhida
    # coeffs = pywt.coeffs_to_array(coefficients[sample_index])

    # Plote os coeficientes wavelet
    plt.figure(figsize=(12, 6))

    # Coeficientes de aproximação
    plt.subplot(level + 2, 1, 1)
    plt.plot(coeffs[:len(coeffs)//2], label='Aproximação (cA)')
    plt.title('Coeficientes de Aproximação')
    plt.legend()

    # Coeficientes de detalhe
    for i in range(level):
        plt.subplot(level + 2, 1, i + 2)
        plt.plot(coeffs[len(coeffs)//2:(len(coeffs)//2 + 2 ** (level - i))], label=f'Detalhe (cD{level - i})')
        plt.title(f'Coeficientes de Detalhe (cD{level - i})')
        plt.legend()

    # Sinal reconstruído (opcional)
    reconstructed_signal = pywt.waverec(coeffs, wavelet_name)
    plt.subplot(level + 2, 1, level + 2)
    plt.plot(reconstructed_signal, label='Sinal Reconstruído')
    plt.title('Sinal Reconstruído')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Construir modelos usando os dados no domínio wavelet
wavelet = 'sym8'
level = 3

# Captura dimensões dos dados
n_samples, n_features = X.shape

# # O level máximo de escalonamento que essa configuração de wavelet permite para os dados em questão
max_level = pywt.dwt_max_level(n_features, wavelet)

# Se o level configurado for menor que o máximo possível, podemos prosseguir
if level <= max_level:
    
    
    # Decompor os sinais usando a wavelet escolhida e o nível de decomposição
    print(range(n_samples))
    mat_coeficientes_wavelet = []
    for i in range(n_samples):
        coeffs = pywt.wavedec(X[i, :], wavelet, level=level)
        flattened_coeffs = np.concatenate(coeffs)
        mat_coeficientes_wavelet.append(flattened_coeffs)
    arr_coeficientes_wavelet = np.array(mat_coeficientes_wavelet)

    # plot_wavelet_coeffs(arr_coeficientes_wavelet, 0, wavelet, level)

    # Dividir os dados em treinamento e teste (70% treinamento, 30% teste)
    X_train, X_test, y_train, y_test = train_test_split(arr_coeficientes_wavelet, y, test_size=0.3, random_state=42)

    # # Treinar o modelo K-NN
    # knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
    # knn.fit(X_train, y_train)

    # # Testar o modelo
    # y_pred = knn_model.predict(descrX_test)

    # # Gerar a matriz de confusão
    # cm = confusion_matrix(y_test, y_pred)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
    # disp.plot(cmap=plt.cm.Blues)

    # # Métricas de Desempenho
    # precision, recall, accuracy, specificity, f1score, df = calculate_metrics(cm)
    # df

else:
    raise ValueError(f'Não é possível dividir este conjunto de dados em um valor superior a {max_level}. Diminua o level')

