% Combinação de Modelos usando Empilhamento
% Limpa variáveis, fecha figuras e limpa o console
close all; clear all; clc; clear

%% Carrega o banco de dados
% Carrega o conjunto de dados pre-processado do arquivo 'matselecionadav3.mat'
load matselecionadav3;
% Descomente a linha abaixo para habilitar o ranking de seleção de variáveis
% conjunto = matselecionadav3;

% ADASYN: Gera amostras sintéticas para balanceamento de classes
% Recupera as características e rótulos do conjunto de dados
adasyn_features = matselecionadav3(:,1:37);
adasyn_labels = matselecionadav3(:,38);
adasyn_beta = [];
adasyn_kDensity = [];
adasyn_kSMOTE = [];
adasyn_featuresAreNormalized = false;
% Gera amostras sintéticas usando ADASYN
[adasyn_featuresSyn, adasyn_labelsSyn] = ADASYN(adasyn_features, adasyn_labels, adasyn_beta, adasyn_kDensity, adasyn_kSMOTE, adasyn_featuresAreNormalized);
% Combina as amostras sintéticas com o conjunto de dados original
adasyn = [adasyn_featuresSyn adasyn_labelsSyn];
conjunto = [matselecionadav3; adasyn];

%% Inicialização e Particionamento dos Dados
% Define uma semente aleatória para reprodutibilidade
rng(0); 
% Cria uma partição do conjunto de dados em treino e teste
cv = cvpartition(size(conjunto,1),'HoldOut',0.3);
% Separa o conjunto de treino e teste com base na partição
idx = cv.test;
dados_treino = conjunto(~idx,:);
dados_teste = conjunto(idx,:);

%% Definindo Conjuntos de Treino e Teste
% Define os conjuntos de treino
x_treino = dados_treino(:,1:37);
y_treino = dados_treino(:,38);
[m,n] = size(x_treino); % Obtém dimensões do conjunto de treino

% Define os conjuntos de teste
x_teste = dados_teste(:,1:37);
y_teste = dados_teste(:,38);
[o,p] = size(x_teste); % Obtém dimensões do conjunto de teste

%% Vetores com Ranking de Seleção de Variáveis
% Aqui você pode habilitar o ranking de seleção de variáveis usando diferentes métodos (DT, RF ou RFF)
rank = [14 19 17 18 23 20 26 4 36 24 6 1 33 3 27 2 21 28 5 29 9 37 25 34 15 11 22 10 16 8 13 35 31 30]; % RFF

%% Redução de Dimensionalidade
% Define o número de descritores após a redução de dimensionalidade
ndesc = 5;
% Aplica a seleção de descritores nos conjuntos de treino e teste
x_treino = selepred(ndesc, m, x_treino, rank);
x_teste = selepred(ndesc, o, x_teste, rank);

%% Criação de Modelos Individuais
% Aqui são criados diferentes modelos individuais, como Árvores de Decisão, k-NN, SVMs
% com diferentes hiperparâmetros.
% Árvores de Decisão
rng('default')
mls = 1;
mdls{1} = fitctree(x_treino, y_treino, 'MinLeafSize', mls);

rng('default')
mls = 10;
mdls{2} = fitctree(x_treino, y_treino, 'MinLeafSize', mls);

rng('default')
mls = 20;
mdls{3} = fitctree(x_treino, y_treino, 'MinLeafSize', mls);

rng('default')
mls = 30;
mdls{4} = fitctree(x_treino, y_treino, 'MinLeafSize', mls);

% k-NN
rng('default')
NumNeighbors = 1;
mdls{5} = fitcknn(x_treino, y_treino, 'NumNeighbors', NumNeighbors);

rng('default')
NumNeighbors = 3;
mdls{6} = fitcknn(x_treino, y_treino, 'NumNeighbors', NumNeighbors);

rng('default')
NumNeighbors = 5;
mdls{7} = fitcknn(x_treino, y_treino, 'NumNeighbors', NumNeighbors);

rng('default')
NumNeighbors = 7;
mdls{8} = fitcknn(x_treino, y_treino, 'NumNeighbors', NumNeighbors);

% SVMs
rng('default')
kernel = 'polynomial';
t = templateSVM('Standardize', 1, 'KernelFunction', kernel);
mdls{9} = fitcecoc(x_treino, y_treino, 'Coding', 'onevsall', 'Learners', t);

rng('default')
kernel = 'gaussian';
t = templateSVM('Standardize', 1, 'KernelFunction', kernel);
mdls{10} = fitcecoc(x_treino, y_treino, 'Coding', 'onevsall', 'Learners', t);

rng('default')
kernel = 'linear';
t = templateSVM('Standardize', 1, 'KernelFunction', kernel);
mdls{11} = fitcecoc(x_treino, y_treino, 'Coding', 'onevsall', 'Learners', t);

%% Combinação de Modelos Usando Empilhamento
% Aqui, estamos criando um ensemble empilhado usando os modelos individuais.
% Para evitar overfitting, usaremos as pontuações de validação cruzada dos modelos base.

rng('default') % Para reprodutibilidade
N = numel(mdls); % Número de modelos
Scores = zeros(size(x_treino,1), N); % Matriz para armazenar as pontuações dos modelos
cv = cvpartition(y_treino, "KFold", 5); % Partição de validação cruzada
for ii = 1:N
    m = crossval(mdls{ii}, 'cvpartition', cv); % Validação cruzada para o modelo atual
    [~, s] = kfoldPredict(m); % Predições do modelo
    Scores(:, ii) = s(:, m.ClassNames==1); % Armazena as pontuações para a classe positiva
end

%% Criação do Conjunto Empilhado
rng('default') % Para reprodutibilidade
t = templateTree('Reproducible', true);
stckdMdl = fitcensemble(Scores, y_treino, 'OptimizeHyperparameters', 'auto', 'Learners', t, 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'AcquisitionFunctionName', 'expected-improvement-plus'));

%% Avaliação das Predições
% Avalia as predições dos modelos individuais e do ensemble empilhado.
label = [];
score = zeros(size(x_teste, 1), N);
mdlLoss = zeros(1, numel(mdls));
for i = 1:N
    [lbl, s] = predict(mdls{i}, x_teste); % Predições dos modelos individuais
    label = [label, lbl]; % Concatena os rótulos preditos
    score(:, i) = s(:, m.ClassNames==1); % Armazena as pontuações dos modelos
    mdlLoss(i) = mdls{i}.loss(x_teste, y_teste); % Calcula a perda do modelo
end

%% Anexando as predições do ensemble empilhado
[lbl, s] = predict(stckdMdl, score); % Predições do ensemble empilhado
label = [label, lbl]; % Concatena os rótulos preditos
mdlLoss(end+1) = stckdMdl.loss(score, y_teste); % Calcula a perda do ensemble empilhado

score = [score, s(:, 1)];  % Anexa as pontuações do ensemble empilhado

%% Mostrando Métricas de Perda
names = {'Tree1', 'Tree2', 'Tree3', 'Tree4', 'kNN1', 'kNN2', 'kNN3', 'kNN4', 'SVMPolynomial', 'SVMGaussiano', 'SVMLinear', 'StackedEnsemble'};
array2table(mdlLoss, 'VariableNames', names)

%% Matriz de Confusão
% Mostra as matrizes de confusão para cada modelo
figure
c = cell(N+1, 1);
for i = 1:numel(c)
    subplot(3, 4, i)
    c{i} = confusionchart(y_teste, label(:, i));
    title(names{i})
end

% Calcula e mostra métricas (precisão, recall, acurácia, especificidade, F1-score)
matMetricas = zeros(12, 5);
for i = 1:12
    conf = confusionmat(y_teste, label(:, i));
    [precision, recall, accuracy, specificity, f1score] = metricclassv2(conf);
    matMetricas(i, 1) = precision;
    matMetricas(i, 2) = recall;
    matMetricas(i, 3) = accuracy;
    matMetricas(i, 4) = specificity;
    matMetricas(i, 5) = f1score;
end

matMetricas

% Salva as métricas em um arquivo
metricas_comb_5_relieff = matMetricas;
save metricas_comb_5_relieff;

