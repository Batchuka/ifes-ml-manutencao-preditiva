% Combinação de Modelos usando Empilhamento
close all; clear all; clc;

%% Carrega o conjunto de dados
load classes; % Carregando o conjunto de dados
conjunto = classes;

%% Definindo a Reprodutibilidade
rng(0); 

%% Particionamento do conjunto de dados
cv = cvpartition(size(conjunto, 1), 'HoldOut', 0.3);

%% Separando o conjunto de treino e teste
idx = cv.test;
dados_treino = conjunto(~idx, :);
dados_teste = conjunto(idx, :);

%% Definindo conjuntos de treino e teste
x_treino = dados_treino(:, 1:2560);
y_treino = dados_treino(:, 2561);

x_teste = dados_teste(:, 1:2560);
y_teste = dados_teste(:, 2561);

%% Cálculo dos Descritores Estatísticos dos Sinais de Vibração
% Para o conjunto de treino
X = x_treino;
mat_descritores = descr_stat(X);
x_treino = mat_descritores;

% Para o conjunto de teste
X = x_teste;
mat_descritores = descr_stat(X);
x_teste = mat_descritores;

%% Criação dos Modelos a serem Empilhados
% Árvore de Decisão
rng('default')
mls_values = [1, 10, 20, 30];
for i = 1:length(mls_values)
    mls = mls_values(i);
    mdls{i} = fitctree(x_treino, y_treino, 'MinLeafSize', mls);
end

% k-NN
rng('default')
NumNeighbors_values = [1, 3, 5, 7];
start_idx = length(mdls) + 1;
for i = 1:length(NumNeighbors_values)
    NumNeighbors = NumNeighbors_values(i);
    mdls{start_idx + i} = fitcknn(x_treino, y_treino, 'NumNeighbors', NumNeighbors);
end

% SVM
rng('default')
kernel_values = {'polynomial', 'gaussian', 'linear'};
start_idx = length(mdls) + length(NumNeighbors_values) + 1;
for i = 1:length(kernel_values)
    kernel = kernel_values{i};
    t = templateSVM('Standardize', 1, 'KernelFunction', kernel);
    mdls{start_idx + i} = fitcecoc(x_treino, y_treino, 'Coding', 'onevsall', 'Learners', t);
end

% Naive Bayes
start_idx = length(mdls) + length(NumNeighbors_values) + length(kernel_values) + 1;
mdls{start_idx} = fitcnb(x_treino, y_treino);

% Ensemble of Decision Trees
start_idx = length(mdls) + length(NumNeighbors_values) + length(kernel_values) + 2;
mdls{start_idx} = fitcensemble(x_treino, y_treino);

%% Combinação dos Modelos Usando Empilhamento
% Calcula as pontuações de classificação para cada modelo usando validação cruzada k-fold
N = numel(mdls);
Scores = zeros(size(x_treino, 1), N);
cv = cvpartition(y_treino, "KFold", 5);
for ii = 1:N
    m = crossval(mdls{ii}, 'cvpartition', cv);
    [~, s] = kfoldPredict(m);
    Scores(:, ii) = s(:, m.ClassNames == 1);
end

%% Criação do Conjunto Empilhado Treinando nas Pontuações de Classificação
t = templateTree('Reproducible', true);
stckdMdl = fitcensemble(Scores, y_treino, 'OptimizeHyperparameters', 'auto', 'Learners', t, 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'AcquisitionFunctionName', 'expected-improvement-plus'));

%% Comparação de Acurácia de Predição
label = [];
score = zeros(size(x_teste, 1), N);
mdlLoss = zeros(1, numel(mdls));
for i = 1:N
    [lbl, s] = predict(mdls{i}, x_teste);
    label = [label, lbl];
    score(:, i) = s(:, m.ClassNames == 1);
    mdlLoss(i) = mdls{i}.loss(x_teste, y_teste);
end

%% Anexa as Predições do Conjunto Empilhado
[lbl, s] = predict(stckdMdl, score);
label = [label, lbl];
mdlLoss(end + 1) = stckdMdl.loss(score, y_teste);

score = [score, s(:, 1)];

%% Mostra Valores das Perdas
names = {'Tree1', 'Tree10', 'Tree20', 'Tree30', 'kNN1', 'kNN3', 'kNN5', 'kNN7', 'SVMPolynomial', 'SVMGaussian', 'SVMlinear', 'NaiveBayes', 'FitcEnsemble', 'StackedEnsemble'};
array2table(mdlLoss, 'VariableNames', names)

%% Matriz de Confusão
figure
c = cell(N + 1, 1);
for i = 1:numel(c)
    subplot(4, 4, i)
    c{i} = confusionchart(y_teste, label(:, i));
    title(names{i})
end
