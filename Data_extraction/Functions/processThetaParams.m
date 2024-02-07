function [window_theta_params] = processThetaParams(window, JSONinfo)
    %Funcion que recibe una ventana de trafico y devuelve los parametros
    %theta de la regresion polinomica de acuerdo con el orden n:
    n = JSONinfo.n;
    domainFIT = JSONinfo.domainFIT;
    
%     regressiontype = strcat('poly', string(n));
    if(sum(isnan(window)) >= 1) %Si hay 1 NaN o más no se puede hacer fit
        window_theta_params = NaN*ones(1, n+1);
    else
        %Parámetros theta:
        try
            h = polyfit(domainFIT, window, n);
            window_theta_params = flip(h);
        catch e
            window_theta_params = NaN*ones(1, n+1);
        end
    end
end