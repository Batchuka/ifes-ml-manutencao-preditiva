% Decompõe os sinais com os filtros wavelet nos níveis de decomposição especificados.

clear,clc,close all

% MODULO 01: CARREGA DADOS
load mat_vibration

variaveis = mat_vibration(:,1:2560);

y = mat_vibration(:,2561);

% Escolher se os modelos serão construídos com os sinais no domínio do
%tempo ou no domínio wavelet.

wavelet = 'YES';

% CONSTROI OS MODELOS COM OS DADOS NO DOMÍNIO DO TEMPO.

if wavelet == 'NOT'
    
    % Particiona conjunto original em Treino e Teste - 70/30
    cv = cvpartition(y,'HoldOut',0.3); % 30% dos dados para teste e 70 para treino.
    
    % Dados de Treino
    Xtrain = variaveis((cv.training),:);
    ytrain = y((cv.training),:);
    
    % Dados de Teste
    Xtest  = variaveis((cv.test),:);
    ytest  = y((cv.test),:);
    
    % Calcula os descritores estatísticos dos sinais de vibração do
    % conjunto de treino.
    X = Xtrain;
    mat_descritores = descr_stat(X);
    descrXtrain = mat_descritores;

    % Calcula os descritores estatísticos dos sinais de vibração do
    % conjunto de teste.
    X = Xtest;
    mat_descritores = descr_stat(X);
    descrXtest = mat_descritores;
    
   %% Treina Modelo de árvore de decisão
    mdl_tree = fitctree(descrXtrain,ytrain,'MaxNumSplits',50,'SplitCriterion','gdi');

    %% Valida o modelo utilizando Cross-Validation - k Folds
    cv_tree = crossval(mdl_tree,'KFold',5);
    %erro = kfoldLoss(cv_knn);
    %accu = 1-erro;
    %fprintf('Resultados dos Treinos\nErro: %f\nAcurácia: %f\n\n',erro*100,accu*100)
    %% realizar o teste 
    [ypred] = predict(mdl_tree,descrXtest(:,:));

    %% Gerar a matriz de confusão
    mConf = confusionmat(ytest,ypred);
    figure()
    confusionchart(mConf)
    %% Métrica de Desempenho
    % Métricas de Desempenho por Classe por modelo individual construído
    % por cada subconunto de treino.
    [precision,sensibility,accuracy,specificity,f1score] = metricclassv1(mConf)
    
else

    % CONSTROI OS MODELOS COM OS DADOS NO DOMÍNIO WAVELET

    variaveis = mat_vibration(:,1:2560);

    y = mat_vibration(:,2561);

    [numSamp,numVar] = size(variaveis);

    % Decomposiçao da matriz de descritores com a wavelet escolhida e no nível de decomposição definido.
    wavelet = 'db4';
    nivelmax = wmaxlev(numVar,wavelet);
    nivel = 4;

    if nivel<= nivelmax

        for i = 1:numSamp
            [C,L] = wavedec(variaveis(i,:),nivel,wavelet);
            nCoef = length(C);
            %Coef(i,1:nCoef) = C;
            matCoef(i,1:nCoef) = C; 
        end
        
        % Particiona conjunto original em Treino e Teste - 70/30
        cv = cvpartition(y,'HoldOut',0.3);% 30% dos dados pra teste e 70 para treino
        % Dados de Treinamento
        Xtrain = matCoef((cv.training),:);
        ytrain = y((cv.training),:);

        % Dados de Teste
        Xtest  = matCoef((cv.test),:);
        ytest  = y((cv.test),:);

        % Calcula os descritores estatísticos dos sinais de vibração do
        % conjunto de treino.
        X = Xtrain;
        mat_descritores = descr_stat(X);
        descrXtrain = mat_descritores;

        % Calcula os descritores estatísticos dos sinais de vibração do
        % conjunto de teste.
        X = Xtest;
        mat_descritores = descr_stat(X);
        descrXtest = mat_descritores;

        %% Treina Modelo de árvore de decisão
        mdl_tree = fitctree(descrXtrain,ytrain,'MaxNumSplits',50,'SplitCriterion','gdi');

        %% Valida o modelo utilizando Cross-Validation - k Folds
        cv_tree = crossval(mdl_tree,'KFold',5);
        %erro = kfoldLoss(cv_knn);
        %accu = 1-erro;
        %fprintf('Resultados dos Treinos\nErro: %f\nAcurácia: %f\n\n',erro*100,accu*100)
        %% realizar o teste 
        [ypred] = predict(mdl_tree,descrXtest(:,:));

        %% Gerar a matriz de confusão
        mConf = confusionmat(ytest,ypred);
        figure()
        confusionchart(mConf)
        %% Métrica de Desempenho
        % Métricas de Desempenho por Classe por modelo individual construído
        % por cada subconunto de treino.
        [precision,sensibility,accuracy,specificity,f1score] = metricclassv1(mConf)

    else
        
        fprintf('O nível máximo de decomposição é = %d',nivelmax);

        %disp('Não existe esse nível de decomposição.');

    end
      
end