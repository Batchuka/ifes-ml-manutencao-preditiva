% descr_stat: Calcula os descritores estatísticos dos sinais discretos no domínio do tempo.

function mat_descritores = descr_stat(X)
% X: Matriz de sinais de entrada (amostras x variáveis).
% mat_descritores: Matriz de descritores calculados (amostras x descritores).

[numSamp, numVar] = size(X); % Obtém o número de amostras e variáveis.

% Inicializa a matriz para armazenar os descritores calculados.
mat_descritores = zeros(numSamp, 13);

for i = 1:numSamp
    % Calcula os descritores estatísticos para cada amostra i.
    
    % Valor médio.
    Xmean = mean(X(i, :));
    
    % Valor de pico.
    Xpeak = max(abs(X(i, :)));
    
    % Amplitude da raiz quadrada.
    Xroot = (mean(sqrt(abs(X(i, :)))))^2;
    
    % Fator de folga (clearance).
    Xclear = Xpeak / Xroot;
    
    % Curtose.
    Xkurt = kurtosis(X(i, :));
    
    % Fator de impulso.
    Ximp = Xpeak / (mean(abs(X(i, :))));
    
    % Desvio padrão.
    Xdpad = std(X(i, :));
    
    % Skewness.
    Xskew = skewness(X(i, :));
    
    % Valor RMS.
    Xrms = rms(X(i, :));
    
    % Fator de forma.
    Xforma = Xrms / (mean(abs(X(i, :))));
    
    % Fator de crista.
    Xcrest = Xpeak / Xrms;
    
    % Valor de pico a pico.
    Xpeak2peak = peak2peak(X(i, :));
    
    % Root-sum-of-squares.
    Xrssq = rssq(X(i, :));
    
    % Armazena os valores dos descritores na matriz mat_descritores.
    mat_descritores(i, :) = [Xmean, Xpeak, Xroot, Xclear, Xkurt, Ximp, Xdpad, Xskew, Xrms, Xforma, Xcrest, Xpeak2peak, Xrssq];
end

end
