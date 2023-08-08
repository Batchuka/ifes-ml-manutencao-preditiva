## 1. Limpeza e Preparação Inicial:
    A primeira parte do código limpa a área de trabalho, a janela de comando e fecha todas as figuras gráficas abertas.

## 2. Carregamento dos Dados:
    Carrega os dados de vibração a partir do arquivo "mat_vibration.mat". Extrai as variáveis de entrada (características) e os rótulos alvo dos dados carregados.

## 3. Escolha do Domínio:
    Define a variável wavelet como 'YES', indicando que os modelos serão construídos no domínio wavelet.

## 4. Construção dos Modelos:
    Entra no bloco else (pois wavelet não é igual a 'NOT'), indicando que os modelos serão construídos no domínio wavelet.

## 5. Decomposição Wavelet e Construção de Modelos no Domínio Wavelet:
    Verifica se o nível de decomposição nivel é menor ou igual ao nivelmax suportado pela wavelet escolhida (no caso, 'db4'). Realiza a decomposição wavelet para cada série temporal de dados, obtendo os coeficientes wavelet. Divide o conjunto original em Treino (70%) e Teste (30%). Calcula os descritores estatísticos dos coeficientes de treinamento e teste. Treina um modelo Árvore de Decisão usando os descritores de treinamento.

## 6. Validação do Modelo por Cross-Validation:
    Realiza a validação cruzada (k-fold cross-validation) com 5 folds para avaliar o desempenho do modelo treinado.

## 7. Teste do Modelo:
    Faz previsões nos dados de teste usando o modelo Árvore de Decisão treinado.

## 8. Avaliação do Modelo:
    Calcula a matriz de confusão a partir das previsões feitas pelo modelo e os rótulos verdadeiros dos dados de teste. Gera um gráfico de matriz de confusão para visualizar o desempenho do modelo.

## 9. Métricas de Desempenho:
    Calcula várias métricas de desempenho, como precisão, sensibilidade, acurácia, especificidade e F1-score, com base na matriz de confusão.

## 10. Conclusão:
    O código conclui com informações sobre o nível máximo de decomposição suportado se o nível escolhido for maior, indicando que a decomposição não é possível nesse caso.