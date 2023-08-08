% Decompõe os sinais com os filtros wavelet nos níveis de decomposição especificados.

% Limpa a área de trabalho, a janela de comando e fecha todas as figuras gráficas.
clear, clc, close all;

% MODULO 01: CARREGA DADOS
% Carrega os dados de vibração a partir do arquivo "mat_vibration.mat".
load mat_vibration;

% Extrai as variáveis de entrada (características) e os rótulos alvo dos dados carregados.
variaveis = mat_vibration(:, 1:2560);
y = mat_vibration(:, 2561);

% Escolher se os modelos serão construídos com os sinais no domínio do tempo ou no domínio wavelet.
wavelet = 'YES';  % Pode ser 'YES' para domínio wavelet ou 'NOT' para domínio do tempo.

% CONSTROI OS MODELOS COM OS DADOS NO DOMÍNIO DO TEMPO OU WAVELET.
if strcmp(wavelet, 'NOT')
    % Particiona o conjunto original em Treino e Teste - 70/30
    cv = cvpartition(y, 'HoldOut', 0.3); % Divide os dados em treinamento (70%) e teste (30%).
    
    % Separa os dados de treinamento
    Xtrain = variaveis(cv.training, :);
    ytrain = y(cv.training, :);
    
    % Separa os dados de teste
    Xtest = variaveis(cv.test, :);
    ytest = y(cv.test, :);
    
    % Calcula os descritores estatísticos dos sinais de vibração do conjunto de treinamento.
    X = Xtrain;
    mat_descritores = descr_stat(X);
    descrXtrain = mat_descritores;

    % Calcula os descritores estatísticos dos sinais de vibração do conjunto de teste.
    X = Xtest;
    mat_descritores = descr_stat(X);
    descrXtest = mat_descritores;
    
    %% Treina Modelo de Árvore de Decisão
    mdl_tree = fitctree(descrXtrain, ytrain, 'MaxNumSplits', 50, 'SplitCriterion', 'gdi');

    %% Valida o modelo utilizando Cross-Validation - k Folds
    cv_tree = crossval(mdl_tree, 'KFold', 5); % Realiza validação cruzada com 5 folds.

    %% Realiza o teste 
    % Faz previsões nos dados de teste usando o modelo Árvore de Decisão treinado.
    [ypred] = predict(mdl_tree, descrXtest);

    %% Gera a matriz de confusão
    % Calcula a matriz de confusão para avaliar o desempenho do modelo.
    mConf = confusionmat(ytest, ypred);
    figure()
    confusionchart(mConf)
    
    %% Métrica de Desempenho
    % Calcula várias métricas de desempenho (precisão, sensibilidade, etc.) com base na matriz de confusão.
    [precision, sensibility, accuracy, specificity, f1score] = metricclassv1(mConf);
    
else
    % CONSTROI OS MODELOS COM OS DADOS NO DOMÍNIO WAVELET

    % Atualiza variáveis com os dados originais de vibração
    variaveis = mat_vibration(:, 1:2560);
    y = mat_vibration(:, 2561);

    [numSamp, numVar] = size(variaveis);

    % Decomposição da matriz de descritores com a wavelet escolhida e no nível de decomposição definido.
    wavelet = 'db4';
    nivelmax = wmaxlev(numVar, wavelet);
    nivel = 4;

    if nivel <= nivelmax
        for i = 1:numSamp
            [C, ~] = wavedec(variaveis(i, :), nivel, wavelet);
            nCoef = length(C);
            matCoef(i, 1:nCoef) = C; 
        end
        
        % Particiona o conjunto original em Treino e Teste - 70/30
        cv = cvpartition(y, 'HoldOut', 0.3); % Divide os dados em treinamento (70%) e teste (30%).

        % Separa os dados de treinamento
        Xtrain = matCoef(cv.training, :);
        ytrain = y(cv.training, :);

        % Separa os dados de teste
        Xtest = matCoef(cv.test, :);
        ytest = y(cv.test, :);

        % Calcula os descritores estatísticos dos sinais de vibração do conjunto de treinamento.
        X = Xtrain;
        mat_descritores = descr_stat(X);
        descrXtrain = mat_descritores;

        % Calcula os descritores estatísticos dos sinais de vibração do conjunto de teste.
        X = Xtest;
        mat_descritores = descr_stat(X);
        descrXtest = mat_descritores;

        %% Treina Modelo de Árvore de Decisão
        mdl_tree = fitctree(descrXtrain, ytrain, 'MaxNumSplits', 50, 'SplitCriterion', 'gdi');

        %% Valida o modelo utilizando Cross-Validation - k Folds
        cv_tree = crossval(mdl_tree, 'KFold', 5); % Realiza validação cruzada com 5 folds.

        %% Realiza o teste 
        % Faz previsões nos dados de teste usando o modelo Árvore de Decisão treinado.
        [ypred] = predict(mdl_tree, descrXtest);

        %% Gera a matriz de confusão
        % Calcula a matriz de confusão para avaliar o desempenho do modelo.
        mConf = confusionmat(ytest, ypred);
        figure()
        confusionchart(mConf)
        
        %% Métrica de Desempenho
        % Calcula várias métricas de desempenho (precisão, sensibilidade, etc.) com base na matriz de confusão.
        [precision, sensibility, accuracy, specificity, f1score] = metricclassv1(mConf);

    else
        fprintf('O nível máximo de decomposição é = %d', nivelmax);
    end
end
