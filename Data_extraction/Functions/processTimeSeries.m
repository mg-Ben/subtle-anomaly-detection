function [output_theta, output_alpha, JSONinfo] = processTimeSeries(WindowsToSimulate, JSONinfo)
    %Funcion para procesar cada serie temporal:
    %Formato de almacenamiento de los datos:
    %Parámetros theta:
    %   [theta0 theta1 theta2...] serie 1 ventana 1 | [theta0 theta1 theta2...] serie 2 ventana 1
    %   [theta0 theta1 theta2...] serie 1 ventana 2 | [theta0 theta1 theta2...] serie 2 ventana 2
    %   [theta0 theta1 theta2...] serie 1 ventana 3 | [theta0 theta1 theta2...] serie 2 ventana 3
    %   ...
    %   ...
    %   [theta0 theta1 theta2...] serie 1 ventana N | [theta0 theta1 theta2...]
    %   serie 2 ventana N
    %Lo mismo con los parámetros alpha
    labels = JSONinfo.labels;
    n = JSONinfo.n;
    number_time_series = length(labels);
    output_theta = NaN*ones(WindowsToSimulate, (number_time_series-1)*(n+1));
    output_alpha = NaN*ones(WindowsToSimulate, (number_time_series-1)*4);
    %For each time serie (for each week):
    for i=2:length(labels) %(The first label of JSON is the domain vector)
        time_serie = JSONinfo.(strcat("week_", labels{i}));
        %Get the n+1 theta coefs and 4 alpha params of each simulated window for this week:
        [theta_params, alpha_params, JSONinfo] = processWindows(time_serie, WindowsToSimulate, labels{i}, i, number_time_series, JSONinfo);
        %Add it to general matrix:
        output_theta(:, 1+(n+1)*(i-2):(n+1)*(i-1)) = theta_params;
        output_alpha(:, 1+4*(i-2):4*(i-1)) = alpha_params;
    end
    %Increment the number of simulated windows in "WindowsToSimulate":
    JSONinfo.Number_of_simulated_windows = JSONinfo.Number_of_simulated_windows+WindowsToSimulate;
end