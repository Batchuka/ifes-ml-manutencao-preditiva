import numpy as np
from scipy.stats import kurtosis, skew

def calculate_statistics(X):
    """
    Calcula os descritores estatísticos dos sinais discretos no domínio do tempo.
    O objetivo é, como o nome indica, descrever o sinal através de suas propriedades
    estatísticas.
    
    Parâmetros:
    X (numpy array): Matriz de sinais de entrada (amostras x variáveis).
    
    Retorna:
    mat_descritores (numpy array): Matriz de descritores calculados (amostras x descritores).
    """
    numSamp, numVar = X.shape  # Obtém o número de amostras e variáveis.

    # Inicializa a matriz para armazenar os descritores calculados.
    mat_descritores = np.zeros((numSamp, 13))

    # Calcula os descritores estatísticos para cada amostra i.
    for i in range(numSamp):
        
        # Valor médio → Representa o valor médio do sinal elétrico dessa amostra.
        Xmean = np.mean(X[i, :])
        
        # Valor de pico → Indica o valor máximo absoluto no sinal.
        Xpeak = np.max(np.abs(X[i, :]))
        
        # Amplitude da raiz quadrada → Uma medida da amplitude do sinal após a aplicação de uma transformação de raiz quadrada.
        Xroot = (np.mean(np.sqrt(np.abs(X[i, :]))))**2
        
        # Fator de folga (clearance) → Relação entre o valor de pico e a amplitude da raiz quadrada.
        Xclear = Xpeak / Xroot
        
        # Curtose → Medida da forma da distribuição do sinal, indicando se possui caudas pesadas ou leves em relação à distribuição normal.
        Xkurt = kurtosis(X[i, :])
        
        # Fator de impulso → Relação entre o valor de pico e o valor médio do sinal.
        Ximp = Xpeak / (np.mean(np.abs(X[i, :])))
        
        # Desvio padrão → Indica a dispersão dos valores do sinal em torno do valor médio.
        Xdpad = np.std(X[i, :])
        
        # Skewness → Medida de assimetria da distribuição do sinal.
        Xskew = skew(X[i, :])
        
        # Valor RMS → Valor eficaz, representando a amplitude do sinal.
        Xrms = np.sqrt(np.mean(X[i, :]**2))
        
        # Fator de forma → Relação entre o valor RMS e o valor médio do sinal.
        Xforma = Xrms / (np.mean(np.abs(X[i, :])))
        
        # Fator de crista → Relação entre o valor de pico e o valor RMS.
        Xcrest = Xpeak / Xrms
        
        # Valor de pico a pico → Diferença entre o valor máximo e mínimo do sinal.
        Xpeak2peak = np.max(X[i, :]) - np.min(X[i, :])
        
        # Root-sum-of-squares → Uma medida da amplitude global do sinal.
        Xrssq = np.sqrt(np.sum(X[i, :]**2))
        
        # Armazena os valores dos descritores na matriz mat_descritores.
        mat_descritores[i, :] = [Xmean, Xpeak, Xroot, Xclear, Xkurt, Ximp, Xdpad, Xskew, Xrms, Xforma, Xcrest, Xpeak2peak, Xrssq]
    
    return mat_descritores
