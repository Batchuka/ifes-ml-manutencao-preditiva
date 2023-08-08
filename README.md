# Projeto de Classificação de Falhas usando Aprendizado de Máquina

Neste projeto, estamos abordando a tarefa de classificação de falhas por meio do uso de algoritmos de aprendizado de máquina. O objetivo é identificar falhas em rolamentos com base em sinais de vibração. O projeto é dividido em várias partes e scripts, cada um desempenhando um papel específico na análise e na construção dos modelos de classificação.

## Estrutura do Projeto

1. Arquivos de Classificação de Falha:
    
    classifica_falha_knn.m: Este script ou função implementa o algoritmo de classificação KNN (K-Nearest Neighbors), que é usado para classificar as falhas nos rolamentos com base em sinais de vibração.
    
    classifica_falha_svm.m: Este script ou função implementa o algoritmo de classificação SVM (Support Vector Machine), utilizado para a classificação de falhas em rolamentos a partir dos sinais de vibração.
    
    classifica_falha_tree.m: Neste script ou função, é implementado o algoritmo de classificação Árvore de Decisão, que é empregado para identificar as falhas nos rolamentos através dos sinais de vibração.

2. Arquivos de Combinação de Modelos:
    
    combinacao_stacking_v1.m: Este script provavelmente está relacionado à técnica de "stacking", que é uma forma de combinação de modelos. No entanto, recomendo estudar mais sobre o assunto para entender o seu funcionamento exato.
    
    combinacao_stacking.m: Este script também está relacionado à técnica de "stacking" ou aprendizado de conjunto, usado para combinar as previsões de diversos modelos visando melhorar a precisão das previsões.

3. Análise Estatística Descritiva:
    
    descr_stat.m: Neste script ou função, são calculados descritores estatísticos para os sinais de vibração dos rolamentos. Esses descritores resumem características importantes dos sinais e podem ser usados para aprimorar a análise e classificação.

4. Métricas de Desempenho:
    
    metricclassv1.m: Este arquivo contém uma função que calcula métricas de desempenho, como precisão, recall, acurácia, especificidade e F1-score, com base em uma matriz de confusão. Essas métricas são utilizadas para avaliar o desempenho dos modelos de classificação.

5. Arquivos de Dados:
    
    classes.mat: Este arquivo provavelmente contém informações sobre as classes de falhas nos rolamentos, fornecendo rótulos para os sinais de vibração.
    
    vibration.mat: Neste arquivo estão armazenados os sinais de vibração dos rolamentos, que são os dados utilizados para a análise e classificação das falhas.

## Descrição dos Diretórios e Arquivos

1. Classificação de Falha com KNN, SVM e Árvore de Decisão:
    
    Os arquivos classifica_falha_knn.m, classifica_falha_svm.m e classifica_falha_tree.m implementam diferentes algoritmos de classificação para identificar falhas nos rolamentos a partir dos sinais de vibração.

2. Aprendizado por Stacking e Ensemble Learning:
    
    Os arquivos combinacao_stacking_v1.m e combinacao_stacking.m se relacionam com técnicas de combinação de modelos para melhorar o desempenho da classificação.

3. Análise Estatística Descritiva:
    
    O arquivo descr_stat.m calcula descritores estatísticos que resumem características dos sinais de vibração, proporcionando informações úteis para a análise.

4. Métricas de Desempenho:
    
    O arquivo metricclassv1.m fornece uma função que avalia o desempenho dos modelos de classificação usando métricas relevantes.

5. Arquivos de Dados:
    
    Os arquivos classes.mat e vibration.mat contêm informações sobre as classes de falhas e os sinais de vibração, respectivamente.

## Análise dos Sinais de Vibração

O diretório contém detalhes sobre os sinais de vibração obtidos dos rolamentos, tanto para o estado de falha quanto para a normalidade. É importante explorar a estrutura desses sinais