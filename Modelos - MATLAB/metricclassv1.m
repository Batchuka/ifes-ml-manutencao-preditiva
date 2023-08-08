% metricclassv1: Calcula medidas de desempenho dos classificadores como:
% - Precision
% - Recall
% - Accuracy
% - Specificity
% - F1 Score

function [precision, recall, accuracy, specificity, f1score] = metricclassv1(conf)
% conf: Matriz de confusão.
% precision: Precisão média.
% recall: Recall médio.
% accuracy: Acurácia média.
% specificity: Especificidade média.
% f1score: F1 Score médio.

[n] = size(conf, 2); % Obtém o número de classes.

precision = zeros(1, n);
recall = zeros(1, n);
accuracy = zeros(1, n);
specificity = zeros(1, n);
f1score = zeros(1, n);

for indice = 1:n
    tp = conf(indice, indice); % Verdadeiros positivos para a classe indice.
    fn = conf(indice, 1:end ~= indice); % Falsos negativos para a classe indice.
    fp = conf(1:end ~= indice, indice); % Falsos positivos para a classe indice.
    tn = conf(1:end ~= indice, 1:end ~= indice); % Verdadeiros negativos para a classe indice.
    
    % Calcula as métricas para a classe indice.
    precision(indice) = sum(tp) / (sum(tp) + sum(fp));
    recall(indice) = sum(tp) / (sum(tp) + sum(fn));
    accuracy(indice) = (sum(tp) + sum(tn)) / (sum(tp) + sum(fp) + sum(tn) + sum(fn));
    specificity(indice) = sum(tn) / (sum(tn) + sum(fp));
    f1score(indice) = (2 * precision(indice) * recall(indice)) / (precision(indice) + recall(indice));
end

% Calcula as médias das métricas para todas as classes.
precision = mean(precision);
recall = mean(recall);
accuracy = mean(accuracy);
specificity = mean(specificity);
f1score = mean(f1score);

end
