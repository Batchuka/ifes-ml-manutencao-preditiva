## 1. Limpeza inicial:
    clear, clc, close all: Isso limpa a área de trabalho, a janela de comando e fecha todas as figuras gráficas abertas anteriormente. Isso garante que o ambiente esteja limpo e organizado para a execução do código.

## 2. Carregamento dos dados:
    load mat_vibration: Carrega os dados de vibração de um arquivo chamado "mat_vibration.mat".

## 3. Extração de variáveis e rótulos:
    variaveis = mat_vibration(:, 1:2560); e y = mat_vibration(:, 2561);: Extrai as variáveis de entrada (características) e os rótulos alvo dos dados carregados. Os dados de vibração são separados nas variáveis variaveis e y.

## 4. Escolha do domínio (tempo ou wavelet):
    wavelet = 'NOT';: Define se os modelos serão construídos no domínio do tempo ou no domínio wavelet. 'NOT' indica construção no domínio do tempo, enquanto 'YES' indicaria construção no domínio wavelet.

## 5. Construção de modelos no domínio do tempo:

    O bloco if wavelet == 'YES' verifica se a opção selecionada é construir modelos no domínio wavelet. Se for verdadeira, o código segue este caminho.

    Particionamento dos dados: Usa o cvpartition para dividir os dados em conjuntos de treinamento e teste na proporção de 70/30.

    Separação de dados: Os dados são separados em conjuntos de treinamento (Xtrain, ytrain) e teste (Xtest, ytest).

    Cálculo de descritores estatísticos: Calcula os descritores estatísticos das séries temporais de vibração, tanto para os conjuntos de treinamento quanto de teste.

    Treinamento do modelo SVM: Cria um modelo de classificação SVM usando os descritores de treinamento. Aqui, a função fitcecoc é usada para treinar o modelo.

    Validação cruzada: Realiza a validação cruzada do modelo usando crossval com 5 folds.

    Teste do modelo: Faz previsões nos dados de teste usando o modelo SVM treinado.

    Matriz de confusão e métricas de desempenho: Calcula a matriz de confusão para avaliar o desempenho do modelo e, em seguida, calcula várias métricas de desempenho como precisão, sensibilidade, especificidade, acurácia e pontuação F1.

## 6. Construção de modelos no domínio wavelet:

    O bloco else é executado se a opção for construir modelos no domínio wavelet.

    Decomposição wavelet: Para cada série temporal, realiza a decomposição wavelet nos níveis especificados e armazena os coeficientes resultantes.

    Particionamento dos dados: Divide os dados em conjuntos de treinamento e teste, assim como feito no caso do domínio do tempo.

    Cálculo de descritores estatísticos: Calcula os descritores estatísticos dos coeficientes wavelet, tanto para os conjuntos de treinamento quanto de teste.

    Treinamento do modelo SVM: Similar ao passo anterior, cria e treina um modelo SVM usando os descritores de treinamento wavelet.

    Validação cruzada: Realiza validação cruzada do modelo usando 5 folds.

    Teste do modelo: Faz previsões nos dados de teste usando o modelo SVM treinado.

    Matriz de confusão e métricas de desempenho: Calcula a matriz de confusão e as métricas de desempenho para avaliar a qualidade do modelo.

## 7. Encerramento e conclusão:
    Fim do bloco de construção do modelo no domínio wavelet.