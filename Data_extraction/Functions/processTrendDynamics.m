function [JSONinfo] = processTrendDynamics(WindowsToSimulate, JSONinfo)
    %Get the theta and alpha params for each window to simulate of each
    %week time serie:
    fprintf("Processing time series...\n");
    [new_theta_params, new_alpha_params, JSONinfo] = processTimeSeries(WindowsToSimulate, JSONinfo);
    %Append it to JSON object:
    key_TP = strcat(strcat(strcat("TP", string(JSONinfo.Tsventana/60)), '_'), string(JSONinfo.n));
    key_AP = strcat(strcat(strcat("AP", string(JSONinfo.Tsventana/60)), '_'), string(JSONinfo.n));
    JSONinfo.(key_TP) = [JSONinfo.(key_TP); new_theta_params];
    JSONinfo.(key_AP) = [JSONinfo.(key_AP); new_alpha_params];
end