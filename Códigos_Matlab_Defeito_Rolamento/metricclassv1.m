function [precision,recall,accuracy,specificity,f1score] = metricclassv1(conf)
% Função para verificar as medidas de desempenho dos classificadores como:
% - Precision
% - Recall
% - Accuracy
% - Specificity
% - F1 Score
% 
% Sintaxe para usar a função:
% [precision,recall,accuracy,specificity,f1score] = metricclass(conf)
% - 'conf' é a matriz confusão
%   
[n] = size(conf,2);
for indice = 1:n
        tp = conf(indice, indice);
        fn = conf(indice, 1:end~= indice); % errados na horizontal
        fp = conf(1:end ~= indice, indice);% errados na vertical
        tn = conf(1:end ~= indice, 1:end ~= indice); 
        precision(indice) = sum(tp) / (sum(tp) + sum(fp));
        recall(indice)    = sum(tp) / (sum(tp) + sum(fn));
        accuracy(indice)  = (sum(tp)+sum(tn)) / (sum(tp)+sum(fp)+sum(tn)+sum(fn));
        specificity(indice) = sum(tn) / (sum(tn) + sum(fp));
        f1score(indice)    = (2 * precision(indice) * recall(indice)) / (precision(indice) + recall(indice));
end
precision = mean(precision);
recall = mean(recall);
accuracy = mean(accuracy);
specificity = mean(specificity);
f1score = mean(f1score);
end

