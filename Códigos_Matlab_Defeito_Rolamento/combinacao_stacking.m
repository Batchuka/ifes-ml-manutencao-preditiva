% COmbinação de modelos usando Empilhamento
close all; clear all; clc; clear
%% Carrega o banco de dados
%load table_covid19.mat
load matselecionadav3; %carregando o banco de dados
%conjunto= matselecionadav3;
%labels=conjunto.Properties.VariableNames;

adasyn_features                 = matselecionadav3(:,1:37);
adasyn_labels                   = matselecionadav3(:,38); %[labels0  ; labels1  ];
adasyn_beta                     = [];   %let ADASYN choose default
adasyn_kDensity                 = [];   %let ADASYN choose default
adasyn_kSMOTE                   = [];   %let ADASYN choose default
adasyn_featuresAreNormalized    = false;    %false lets ADASYN handle normalization
    
[adasyn_featuresSyn, adasyn_labelsSyn] = ADASYN(adasyn_features, adasyn_labels, adasyn_beta, adasyn_kDensity, adasyn_kSMOTE, adasyn_featuresAreNormalized);

adasyn = [adasyn_featuresSyn adasyn_labelsSyn];

conjunto = [matselecionadav3; adasyn];

%% para reprodutibilidade
rng(0); 
%% Particionamento do conjunto de dados
cv = cvpartition(size(conjunto,1),'HoldOut',0.3);
%% Separando o conjunto de treino e teste
idx = cv.test;
dados_treino = conjunto(~idx,:);
dados_teste = conjunto(idx,:);
%% Definindo conjunto de treino
x_treino= dados_treino(:,1:37);
y_treino= dados_treino(:,38);
[m,n]=size(x_treino); % retornar dimensões do conjunto de teste m - linhas, n colunas
%% Definindo conjunto de teste
x_teste= dados_teste(:,1:37);
y_teste= dados_teste(:,38);
[o,p]=size(x_teste); % retornar dimensões do conjunto de teste o - linhas, p colunas
%% Vetores com ranking de seleção de variáveis
% Habilitar o ranking de seleção de variáveis pelo método de seleção
%(DT - árvore de decisão/RF - floresta aleatoria / RFF - Relief-F)
%rank=[9 12 6 4 1 26 14 33 2 16 29 7 36 23 15 3 31 13 24 19 22 35 5 21 30 10 8 32 27 11 25 17 20 28 18 34 37]; %DT
%rank=[12 9 4 6 18 23 26 2 14 1 32 17 33 37 15 21 30 34 35 22 5 29 16 7 13 19 28 25 27 11 10 24 20 31 3 8 36]; %RF
rank=[14 19 17 18 23 20 26 4 36 24 6 1 33 3 27 2 21 28 5 29 9 37 25 34 15 11 22 10 16 8 13 35 31 30]; %RFF
%% Definir número de descritores para o banco de dados
ndesc = 5; % Número de descritores
x_treino=selepred(ndesc,m,x_treino, rank);
x_teste=selepred(ndesc,o,x_teste, rank);
%% Criar os modelos a serem empilhados 
%% Árvore de Decisão
% rng('default')
% mdls{3} = fitctree(x_treino,y_treino);
rng('default')
mls=1;
mdls{1}=fitctree(x_treino,y_treino,'MinLeafSize',mls);
%[ecg_class] = predict(modelo_dt,x_teste1);  % realizar o teste
%ConfMat = confusionmat(y_teste,ecg_class); % Gerar a matriz de confusão
% confusionchart(y_teste,ecg_class);
%[precision,recall,accuracy,specificity,f1score] = metricclassv1(ConfMat); % Aplicar as métricas
%x1=[precision,recall,accuracy,specificity,f1score];
 
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
%% Floresta Aleatória
% rng('default')
% num_trees = 10; 
% Method = 'classification';
% mdls{12}=TreeBagger(num_trees,x_treino,y_treino,'Method','classification','OOBPrediction','on'); %Cria o modelo treebagger

%err1 = oobError(modelo_treebagger);
%ecg_class = predict(modelo_treebagger,x_teste1);   %Predição do classificador
%classe=str2double(ecg_class);
%ConfMat = confusionmat(y_teste,classe); %Gera a matriz de confusão
%figure ()
%confusionchart(y_teste,classe); %Imprime a matriz de confusão
%[precision,recall,accuracy,specificity,f1score] = metricclassv1(ConfMat); % Aplicar as métricas
%x1=[precision,recall,accuracy,specificity,f1score];

% rng('default')
% num_trees = 20; 
% Method = 'classification';
% rf_class=TreeBagger(num_trees,x_treino,y_treino,'Method','classification','OOBPrediction','on'); %Cria o modelo treebagger
% mdls{13}=str2double(rf_class);
%  
% rng('default')
% num_trees = 50; 
% Method = 'classification';
% rf_class=TreeBagger(num_trees,x_treino,y_treino,'Method','classification','OOBPrediction','on'); %Cria o modelo treebagger
% mdls{14}=str2double(rf_class);
%   
% rng('default')
% num_trees = 100; 
% Method = 'classification';
% rf_class=TreeBagger(num_trees,x_treino,y_treino,'Method','classification','OOBPrediction','on'); %Cria o modelo treebagger
% mdls{15}=str2double(rf_class);
%   
% rng('default')
% num_trees = 200; 
% Method = 'classification';
% rf_class=TreeBagger(num_trees,x_treino,y_treino,'Method','classification','OOBPrediction','on'); %Cria o modelo treebagger
% mdls{16}=str2double(rf_class);
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

% stckdMdl = fitcensemble(Scores,y_treino,
%    'OptimizeHyperparameters','auto', ...
%    'Learners',t, ...
%    'HyperparameterOptimizationOptions',struct('Verbose',0,'AcquisitionFunctionName','expected-improvement-plus'));
%% Compara acurácia de predição
%Find the predicted labels, scores, and loss values of the test data set for the base models and the stacked ensemble.
label = [];
score = zeros(size(x_teste,1),N);
mdlLoss = zeros(1,numel(mdls));
for i = 1:N
    [lbl,s] = predict(mdls{i},x_teste);
    label = [label,lbl];
    score(:,i) = s(:,m.ClassNames==1);
    mdlLoss(i) = mdls{i}.loss(x_teste, y_teste);
end

%% Attach the predictions from the stacked ensemble to label and mdlLoss.
[lbl,s] = predict(stckdMdl,score);
label = [label,lbl];
mdlLoss(end+1) = stckdMdl.loss(score,y_teste);

score = [score,s(:,1)];  %Concatenate the score of the stacked ensemble to the scores of the base models.

% %% Mostrar valores das perdas 
names = {'Tree1','Tree2','Tree3','Tree4','kNN1','kNN2','kNN3','kNN4','SVMPolynomial','SVMGaussiano','SVMLinear','StackedEnsemble'};
array2table(mdlLoss,'VariableNames',names)

%% Matriz de confusão
figure
c = cell(N+1,1);
for i = 1:numel(c)
    subplot(3,4,i)
    c{i} = confusionchart(y_teste,label(:,i));
    title(names{i})
end

matMetricas = zeros (12, 5);
for i = 1:12
    conf = confusionmat(y_teste,label(:,i));
    [precision,recall,accuracy,specificity,f1score] = metricclassv2(conf);
    matMetricas(i,1) = precision;
    matMetricas(i,2) = recall;
    matMetricas(i,3) = accuracy;
    matMetricas(i,4) = specificity;
    matMetricas(i,5) = f1score;
end

matMetricas

metricas_comb_5_relieff = matMetricas;

save metricas_comb_5_relieff;
