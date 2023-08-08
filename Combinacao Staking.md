## 1. Inicialização e Carregamento de Dados:
    Inicializa o ambiente e carrega o conjunto de dados "classes".

## 2. Particionamento do Conjunto de Dados:
    Usa o método cvpartition para criar uma partição do conjunto de dados, separando-o em treino e teste. O tamanho do conjunto de teste é definido com 30% do tamanho total.

## 3. Separação dos Dados de Treino e Teste:
    Usa os índices da partição para separar os dados em conjuntos de treino e teste.

## 4. Definição dos Conjuntos de Treino e Teste:
    Define x_treino e y_treino como as características e rótulos dos dados de treino, respectivamente. Define x_teste e y_teste como as características e rótulos dos dados de teste, respectivamente.

## 5. Cálculo dos Descritores Estatísticos dos Sinais de Vibração:
    Calcula os descritores estatísticos para os conjuntos de treino e teste usando a função descr_stat. Isso transforma as séries temporais de vibração em vetores de características estatísticas.

## 6. Criação de Modelos Individuais:
    Cria diversos modelos individuais, como Árvores de Decisão, k-NN, SVMs, Naive Bayes e um Ensemble de Árvores de Decisão. Utiliza diferentes hiperparâmetros para configurar cada modelo (por exemplo, MinLeafSize para Árvores de Decisão e NumNeighbors para k-NN).

## 7. Combinação dos Modelos usando Empilhamento:
    Calcula as pontuações de classificação para cada modelo usando validação cruzada k-fold. As pontuações são armazenadas em uma matriz Scores, onde cada coluna representa as pontuações de um modelo diferente.

## 8. Criação do Conjunto Empilhado usando as Pontuações de Classificação:
    Cria um modelo de Ensemble de Árvores de Decisão usando as pontuações de classificação dos modelos individuais. Utiliza a função fitcensemble para treinar o conjunto empilhado, otimizando seus hiperparâmetros automaticamente.

## 9. Comparação da Acurácia de Predição:
    Faz previsões para o conjunto de teste usando cada um dos modelos individuais. Calcula a perda (loss) para cada modelo individual. Armazena os resultados de previsões e perdas para análise posterior.

## 10. Anexação das Predições do Conjunto Empilhado:
    Faz previsões usando o conjunto empilhado para as pontuações de classificação dos modelos individuais. Concatena essas previsões às previsões dos modelos individuais anteriores.

## 11. Mostrando Valores das Perdas:
    Cria uma tabela com os valores de perda para cada modelo individual e o conjunto empilhado. Os nomes dos modelos são listados na tabela para referência.

## 12. Matriz de Confusão:
    Cria uma matriz de gráficos de confusão para cada modelo individual e o conjunto empilhado. Cada gráfico de confusão mostra visualmente o desempenho das previsões em relação aos rótulos reais.