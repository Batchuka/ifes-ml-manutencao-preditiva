% Calcula os valores dos descritores estatísticos dos sinais discretos no domínio do
% tempo. 

function mat_descritores = descr_stat(X);

[numSamp,numVar] = size(X);

for i = 1:numSamp
    % Valor médio.
    Xmean = mean(X(i,:));
    
    % Valor de pico.
    Xpeak = max(abs(X(i,:)));
    
    % Amplitude da raiz quadrada
    Xroot = (mean(sqrt(abs(X(i,:)))))^2;
    
    % Fator de folga (clearence).
    Xclear = Xpeak/Xroot;
    
    % Curtose.
    Xkurt = kurtosis(X(i,:));
    
    % Fator de impulso.
    Ximp = Xpeak/(mean(abs(X(i,:))));
    
    % Desvio padrão.
    Xdpad = std(X(i,:));
    
    % Skewness.
    Xskew = skewness(X(i,:));
    
    % Valor RMS.
    Xrms = rms(X(i,:));
    
    % Fator de forma.
    Xforma = Xrms/(mean(abs(X(i,:))));
    
    % Fator de crista.
    Xcrest = Xpeak/Xrms;
    
    % Valor de pico a pico.
    Xpeak2peak = peak2peak(X(i,:));
    
    % Root-sum-of-squares.
    Xrssq = rssq(X(i,:));
    
    mat_descritores(i,:) = [Xmean Xpeak Xroot Xclear Xkurt Ximp Xdpad Xskew Xrms Xforma Xcrest Xpeak2peak Xrssq];
end

end

    
    
    
   
    
   
    
    