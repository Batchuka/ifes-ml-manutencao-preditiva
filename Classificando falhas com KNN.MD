## 1. Limpeza e Preparação Inicial:
    clear, clc, close all: Isso limpa variáveis da área de trabalho, limpa a janela de comando e fecha todas as figuras gráficas abertas.

## 2. Carregamento dos Dados:
    load mat_vibration: Carrega os dados de vibração a partir do arquivo "mat_vibration.mat". Supõe-se que o arquivo contenha uma matriz onde as colunas 1 a 2560 representam as variáveis de entrada (características) e a coluna 2561 representa os rótulos alvo.

## 3. Separação dos Dados:
    variaveis = mat_vibration(:, 1:2560); Extrai as colunas de 1 a 2560 (variáveis) dos dados carregados. y = mat_vibration(:, 2561); Extrai a coluna 2561 (rótulos alvo) dos dados carregados.

## 4. Escolha do Domínio de Modelagem:
    wavelet = 'NOT'; Define se os modelos serão construídos usando os dados no domínio do tempo ("NOT" indica "não wavelet").

## 5. Modelagem no Domínio do Tempo:
    O bloco "if wavelet == 'NOT'" é acionado se a variável wavelet for 'NOT', ou seja, se a modelagem for feita no domínio do tempo. cv = cvpartition(y, 'HoldOut', 0.3); Divide os dados em conjuntos de treinamento e teste usando validação cruzada hold-out com uma divisão de 70% para treinamento e 30% para teste.

## 6. Separação dos Dados de Treinamento e Teste:
    Os dados são divididos com base na partição criada usando a função cvpartition.

## 7. Cálculo de Descritores Estatísticos:
    descr_stat: Presumivelmente, uma função personalizada que calcula os descritores estatísticos dos sinais de vibração. Os descritores são calculados para os conjuntos de treinamento e teste.

## 8. Treinamento do Modelo k-NN:
    mdl_knn = fitcknn(descrXtrain, ytrain, ...); Cria um modelo de classificação k-NN usando os descritores estatísticos do conjunto de treinamento.

## 9. Validação Cruzada k-Folds:
    cv_knn = crossval(mdl_knn, 'KFold', 5); Realiza validação cruzada k-Folds (5 folds) para avaliar o desempenho do modelo k-NN.

## 10. Teste do Modelo e Matriz de Confusão:
    [ypred] = predict(mdl_knn, descrXtest(:, :)); Usa o modelo k-NN treinado para fazer previsões nos dados de teste. mConf = confusionmat(ytest, ypred); Calcula a matriz de confusão comparando os rótulos verdadeiros (ytest) com as previsões (ypred).

## 11. Visualização da Matriz de Confusão:
    figure(); confusionchart(mConf); Gera um gráfico de matriz de confusão para visualizar o desempenho do modelo.

## 12. Cálculo de Métricas de Desempenho:
    [precision, sensibility, accuracy, specificity, f1score] = metricclassv1(mConf); Calcula várias métricas de desempenho, como precisão, sensibilidade, acurácia, especificidade e pontuação F1, com base na matriz de confusão.

## 13. Modelagem no Domínio Wavelet (caso contrário):
    Se a variável wavelet não for 'NOT', ou seja, se a modelagem for no domínio wavelet, esse bloco de código será executado. Realiza uma decomposição wavelet dos dados de vibração.

## 14. Continuação do Processo de Modelagem (semelhante à etapa 6 até 12):
    A modelagem, treinamento do modelo k-NN, validação cruzada, teste do modelo, geração de matriz de confusão e cálculo de métricas de desempenho são realizados no domínio wavelet.

## 15. Justificativas:
    O código carrega dados de vibração, permite a escolha entre modelagem no domínio do tempo e no domínio wavelet e, em seguida, conduz a construção, treinamento e avaliação de modelos k-NN nos dois domínios. As métricas de desempenho são calculadas para avaliar quão bem os modelos se saem na classificação dos dados de teste. A escolha entre domínio do tempo e domínio wavelet pode ser útil para explorar diferentes abordagens de modelagem e avaliar qual delas é mais eficaz para o problema em questão.