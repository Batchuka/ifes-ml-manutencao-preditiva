% Este código decompõe sinais com filtros wavelet nos níveis de decomposição especificados.

clear, clc, close all

% MÓDULO 01: CARREGA DADOS
load mat_vibration

% Separa as variáveis e o rótulo
variaveis = mat_vibration(:, 1:2560);
y = mat_vibration(:, 2561);

% Escolha se os modelos serão construídos com os sinais no domínio do tempo ou no domínio wavelet.
wavelet = 'NOT';

% CONSTRÓI OS MODELOS COM OS DADOS NO DOMÍNIO DO TEMPO.
if wavelet == 'NOT'
    
    % Particiona o conjunto original em Treino e Teste - 70/30
    cv = cvpartition(y, 'HoldOut', 0.3); % 30 dos dados para teste e 70 para treino.
    
    % Separa os dados de treino
    Xtrain = variaveis((cv.training), :);
    ytrain = y((cv.training), :);
    
    % Separa os dados de teste
    Xtest  = variaveis((cv.test), :);
    ytest  = y((cv.test), :);
    
    % Calcula os descritores estatísticos dos sinais de vibração do conjunto de treino.
    X = Xtrain;
    mat_descritores = descr_stat(X);
    descrXtrain = mat_descritores;

    % Calcula os descritores estatísticos dos sinais de vibração do conjunto de teste.
    X = Xtest;
    mat_descritores = descr_stat(X);
    descrXtest = mat_descritores;
    
    %% Treina o modelo knn
    mdl_knn = fitcknn(descrXtrain, ytrain, 'Distance', 'euclidean', 'NumNeighbors', 3);

    %% Valida o modelo utilizando Cross-Validation - k Folds
    cv_knn = crossval(mdl_knn, 'KFold', 5);
    
    %% Realiza o teste 
    [ypred] = predict(mdl_knn, descrXtest(:, :));

    %% Gera a matriz de confusão
    mConf = confusionmat(ytest, ypred);
    figure()
    confusionchart(mConf)
    
    %% Métrica de Desempenho
    % Métricas de Desempenho por Classe por modelo individual construído por cada subconjunto de treino.
    [precision, sensibility, accuracy, specificity, f1score] = metricclassv1(mConf)
else
    % CONSTRÓI OS MODELOS COM OS DADOS NO DOMÍNIO WAVELET

    variaveis = mat_vibration(:, 1:2560);

    y = mat_vibration(:, 2561);

    [numSamp, numVar] = size(variaveis);

    % Decomposição da matriz de descritores com a wavelet escolhida e no nível de decomposição definido.
    wavelet = 'sym8';
    nivelmax = wmaxlev(numVar, wavelet);
    nivel = 3;

    if nivel <= nivelmax

        for i = 1:numSamp
            [C, L] = wavedec(variaveis(i,:), nivel, wavelet);
            nCoef = length(C);
            matCoef(i, 1:nCoef) = C; 
        end
        
        % Particiona o conjunto original em Treino e Teste - 70/30
        cv = cvpartition(y,'HoldOut',0.3);% 30% dos dados pra teste e 70 para treino
        
        % Separa os dados de treinamento
        Xtrain = matCoef((cv.training), :);
        ytrain = y((cv.training), :);

        % Separa os dados de teste
        Xtest  = matCoef((cv.test), :);
        ytest  = y((cv.test), :);

        % Calcula os descritores estatísticos dos sinais de vibração do conjunto de treino.
        X = Xtrain;
        mat_descritores = descr_stat(X);
        descrXtrain = mat_descritores;

        % Calcula os descritores estatísticos dos sinais de vibração do conjunto de teste.
        X = Xtest;
        mat_descritores = descr_stat(X);
        descrXtest = mat_descritores;

        %% Treina o modelo knn
        mdl_knn = fitcknn(descrXtrain,ytrain,'Distance','euclidean','NumNeighbors',3);

        %% Valida o modelo utilizando Cross-Validation - k Folds
        cv_knn = crossval(mdl_knn,'KFold',5);
        
        %% Realiza o teste 
        [ypred] = predict(mdl_knn,descrXtest(:,:));

        %% Gera a matriz de confusão
        mConf = confusionmat(ytest,ypred);
        figure()
        confusionchart(mConf)
        
        %% Métrica de Desempenho
        % Métricas de Desempenho por Classe por modelo individual construído por cada subconjunto de treino.
        [precision, sensibility, accuracy, specificity, f1score] = metricclassv1(mConf)
    end
end