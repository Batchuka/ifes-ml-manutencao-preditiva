% COmbinação de modelos usando empilhamento
close all; clear all; clc;

%% Carrega o conjunto de dados
load classes; %carregando o conjunto de dados
conjunto = classes;

%% para reprodutibilidade
rng(0); 

%% Particionamento do conjunto de dados
cv = cvpartition(size(conjunto,1),'HoldOut',0.3);

%% Separando o conjunto de treino e teste
idx = cv.test;
dados_treino = conjunto(~idx,:);
dados_teste = conjunto(idx,:);

%% Definindo conjunto de treino
x_treino= dados_treino(:,1:2560);
y_treino= dados_treino(:,2561);

%% Definindo conjunto de teste
x_teste= dados_teste(:,1:2560);
y_teste= dados_teste(:,2561);

% Calcula os descritores estatísticos dos sinais de vibração do
    % conjunto de treino.
    X = x_treino;
    mat_descritores = descr_stat(X);
    x_treino = mat_descritores;

    % Calcula os descritores estatísticos dos sinais de vibração do
    % conjunto de teste.
    X = x_teste;
    mat_descritores = descr_stat(X);
    x_teste = mat_descritores;


%head(dados_treino)
%% Criar os modelos a serem empilhados
%% Árvore de Decisão
rng('default')
mls=1;
mdls{1}=fitctree(x_treino,y_treino,'MinLeafSize',mls);
 
rng('default')
mls=10;
mdls{2}=fitctree(x_treino,y_treino,'MinLeafSize',mls);

rng('default')
mls=20;
mdls{3}=fitctree(x_treino,y_treino,'MinLeafSize',mls);

rng('default')
mls=30;
mdls{4}=fitctree(x_treino,y_treino,'MinLeafSize',mls);

%% k-NN
rng('default')
NumNeighbors = 1;
mdls{5}=fitcknn(x_treino,y_treino,'NumNeighbors',NumNeighbors); %,'Distance','cosine');

rng('default')
NumNeighbors = 3;
mdls{6}=fitcknn(x_treino,y_treino,'NumNeighbors',NumNeighbors); %,'Distance','cosine');

rng('default')
NumNeighbors = 5;
mdls{7}=fitcknn(x_treino,y_treino,'NumNeighbors',NumNeighbors); %,'Distance','cosine');

rng('default')
NumNeighbors = 7;
mdls{8}=fitcknn(x_treino,y_treino,'NumNeighbors',NumNeighbors); %,'Distance','cosine');

%% SVM
rng('default')
kernel='polynomial';              
t = templateSVM('Standardize',1,'KernelFunction',kernel);     
mdls{9} = fitcecoc(x_treino,y_treino,'Coding','onevsall','Learners',t); 

rng('default')
kernel='gaussian';              
t = templateSVM('Standardize',1,'KernelFunction',kernel);     
mdls{10} = fitcecoc(x_treino,y_treino,'Coding','onevsall','Learners',t);

rng('default')
kernel='linear';              
t = templateSVM('Standardize',1,'KernelFunction',kernel);     
mdls{11} = fitcecoc(x_treino,y_treino,'Coding','onevsall','Learners',t);

% Naive Bayes
rng('default')
mdls{12} = fitcnb(x_treino,y_treino);

% Ensemble of decision trees
rng('default')
mdls{13} = fitcensemble(x_treino,y_treino);

%% Combina Modelos Usando Empilhamento
%If you use only the prediction scores of the base models on the training data, 
%the stacked ensemble might be subject to overfitting. To reduce overfitting, use the k-fold cross-validated scores instead. 
%To ensure that you train each model using the same k-fold data split, create a cvpartition object and pass that object to the 
%crossval function of each base model. This example is a binary classification problem, so you only need to consider scores
%for either the positive or negative class.

rng('default') % For reproducibility
N = numel(mdls);
Scores = zeros(size(x_treino,1),N);
cv = cvpartition(y_treino,"KFold",5);
for ii = 1:N
    m = crossval(mdls{ii},'cvpartition',cv);
    [~,s] = kfoldPredict(m);
    Scores(:,ii) = s(:,m.ClassNames==1);
end

%% Cria o conjunto empilhado treinando-o nas pontuações de classificação com validação cruzada 
rng('default') % For reproducibility
t = templateTree('Reproducible',true);
%t = templateTree('MaxNumCategories',2);
stckdMdl = fitcensemble(Scores,y_treino,'OptimizeHyperparameters','auto','Learners',t,'HyperparameterOptimizationOptions',struct('Verbose',0,'AcquisitionFunctionName','expected-improvement-plus'));

%stckdMdl = fitcensemble(Scores,y_treino,
 %   'OptimizeHyperparameters','auto', ...
  %  'Learners',t, ...
   % 'HyperparameterOptimizationOptions',struct('Verbose',0,'AcquisitionFunctionName','expected-improvement-plus'));

%% Compara acurácia de predição
%Find the predicted labels, scores, and loss values of the test data set for the base models and the stacked ensemble.
label = [];
score = zeros(size(x_teste,1),N);
mdlLoss = zeros(1,numel(mdls));
for i = 1:N
    [lbl,s] = predict(mdls{i},x_teste);
    label = [label,lbl];
    score(:,i) = s(:,m.ClassNames==1);
    mdlLoss(i) = mdls{i}.loss(x_teste,y_teste);
    %L = loss(Mdl,X,Y) 
end

%% Attach the predictions from the stacked ensemble to label and mdlLoss.
[lbl,s] = predict(stckdMdl,score);
label = [label,lbl];
mdlLoss(end+1) = stckdMdl.loss(score,y_teste);

score = [score,s(:,1)];  %Concatenate the score of the stacked ensemble to the scores of the base models.

%% Mostrar valores das perdas 
names = {'Tree1','Tree10','Tree20','Tree30','kNN1','kNN3','kNN5','kNN7','SVMPolynomial','SVMGaussian','SVMlinear',...
    'NaiveBayes','FitcEnsemble','StackedEnsemble'};
array2table(mdlLoss,'VariableNames',names)

%% Matriz de confusão
figure
c = cell(N+1,1);
for i = 1:numel(c)
    subplot(4,4,i)
    c{i} = confusionchart(y_teste,label(:,i));
    title(names{i})
end