function [window_alpha_params] = processAlphaParams(window, JSONinfo)
    %Funcion que recibe una ventana de trafico y devuelve los parametros
    %alpha-stable eliminando la tendencia precisa de orden 9:
    Tsventana = JSONinfo.Tsventana;
    
    if(sum(isnan(window)) >= 1) %Si hay 1 NaN o más no se puede hacer fit
        window_alpha_params = NaN*ones(1, 4);
    else
        %Parámetros alpha:
        h_accurate = fit([1:Tsventana]', window, 'poly9');
        %Get alpha-stable remaining part:
        Xt = window-h_accurate(1:Tsventana)';
        try
            Parametros_AlphaStable = fitdist(Xt', 'Stable');
            alpha = Parametros_AlphaStable.alpha;
            beta = Parametros_AlphaStable.beta;
            gamma = Parametros_AlphaStable.gam;
            delta = Parametros_AlphaStable.delta;
        catch
            alpha = NaN; beta = NaN; gamma = NaN; delta = NaN;
        end
        window_alpha_params = [alpha, beta, gamma, delta];
    end
end