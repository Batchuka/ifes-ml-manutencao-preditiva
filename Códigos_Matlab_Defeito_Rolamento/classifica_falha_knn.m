% Decomp�e os sinais com os filtros wavelet nos n�veis de decomposi��o especificados.

clear,clc,close all

% MODULO 01: CARREGA DADOS
load mat_vibration

variaveis = mat_vibration(:,1:2560);

y = mat_vibration(:,2561);

% Escolher se os modelos ser�o constru�dos com os sinais no dom�nio do
%tempo ou no dom�nio wavelet.

wavelet = 'NOT';

% CONSTROI OS MODELOS COM OS DADOS NO DOM�NIO DO TEMPO.

if wavelet == 'NOT'
    
    % Particiona conjunto original em Treino e Teste - 70/30
    cv = cvpartition(y,'HoldOut',0.3); % 30% dos dados para teste e 70 para treino.
    
    % Dados de Treino
    Xtrain = variaveis((cv.training),:);
    ytrain = y((cv.training),:);
    
    % Dados de Teste
    Xtest  = variaveis((cv.test),:);
    ytest  = y((cv.test),:);
    
    % Calcula os descritores estat�sticos dos sinais de vibra��o do
    % conjunto de treino.
    X = Xtrain;
    mat_descritores = descr_stat(X);
    descrXtrain = mat_descritores;

    % Calcula os descritores estat�sticos dos sinais de vibra��o do
    % conjunto de teste.
    X = Xtest;
    mat_descritores = descr_stat(X);
    descrXtest = mat_descritores;
    
    %% Treina Modelo knn
    mdl_knn = fitcknn(descrXtrain,ytrain,'Distance','euclidean','NumNeighbors',3);

    %% Valida o modelo utilizando Cross-Validation - k Folds
    cv_knn = crossval(mdl_knn,'KFold',5);
    %erro = kfoldLoss(cv_knn);
    %accu = 1-erro;
    %fprintf('Resultados dos Treinos\nErro: %f\nAcur�cia: %f\n\n',erro*100,accu*100)
    %% realizar o teste 
    [ypred] = predict(mdl_knn,descrXtest(:,:));

    %% Gerar a matriz de confus�o
    mConf = confusionmat(ytest,ypred);
    figure()
    confusionchart(mConf)
    %% M�trica de Desempenho
    % M�tricas de Desempenho por Classe por modelo individual constru�do
    % por cada subconunto de treino.
    [precision,sensibility,accuracy,specificity,f1score] = metricclassv1(mConf)
    
else

    % CONSTROI OS MODELOS COM OS DADOS NO DOM�NIO WAVELET

    variaveis = mat_vibration(:,1:2560);

    y = mat_vibration(:,2561);

    [numSamp,numVar] = size(variaveis);

    % Decomposi�ao da matriz de descritores com a wavelet escolhida e no n�vel de decomposi��o definido.
    wavelet = 'sym8';
    nivelmax = wmaxlev(numVar,wavelet);
    nivel = 3;

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

        % Calcula os descritores estat�sticos dos sinais de vibra��o do
        % conjunto de treino.
        X = Xtrain;
        mat_descritores = descr_stat(X);
        descrXtrain = mat_descritores;

        % Calcula os descritores estat�sticos dos sinais de vibra��o do
        % conjunto de teste.
        X = Xtest;
        mat_descritores = descr_stat(X);
        descrXtest = mat_descritores;

        %% Treina Modelo knn
        mdl_knn = fitcknn(descrXtrain,ytrain,'Distance','euclidean','NumNeighbors',3);

        %% Valida o modelo utilizando Cross-Validation - k Folds
        cv_knn = crossval(mdl_knn,'KFold',5);
        %erro = kfoldLoss(cv_knn);
        %accu = 1-erro;
        %fprintf('Resultados dos Treinos\nErro: %f\nAcur�cia: %f\n\n',erro*100,accu*100)
        %% realizar o teste 
        [ypred] = predict(mdl_knn,descrXtest(:,:));

        %% Gerar a matriz de confus�o
        mConf = confusionmat(ytest,ypred);
        figure()
        confusionchart(mConf)
        %% M�trica de Desempenho
        % M�tricas de Desempenho por Classe por modelo individual constru�do
        % por cada subconunto de treino.
        [precision,sensibility,accuracy,specificity,f1score] = metricclassv1(mConf)

    else
        
        fprintf('O n�vel m�ximo de decomposi��o � = %d',nivelmax);

    end
      
end