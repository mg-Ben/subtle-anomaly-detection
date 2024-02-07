function [output_theta, output_alpha, JSONinfo] = processWindows(time_serie, WindowsToSimulate, label, index_time_serie, number_time_series, JSONinfo)
    %Funcion para procesar por ventanas una serie temporal concreta
    key_TP = strcat(strcat(strcat("TP", string(JSONinfo.Tsventana/60)), '_'), string(JSONinfo.n));
    window_checkpoint = size(JSONinfo.(key_TP), 1)+1;
    Tsventana = JSONinfo.Tsventana;
    output_theta = NaN*ones(WindowsToSimulate, JSONinfo.n+1);
    output_alpha = NaN*ones(WindowsToSimulate, 4);
    printStatusEach_X_windows = 200;
    %Process each window:
    index=1;
    for i_window=window_checkpoint:WindowsToSimulate+window_checkpoint-1
        %Print processing status:
        if rem(index-1, printStatusEach_X_windows) == 0
            fprintf("[%d/%d] Processing windows for %s time serie -> %.4f%%...\n", index_time_serie, number_time_series, label, index*100/WindowsToSimulate);
        end
        window = time_serie(i_window:i_window+Tsventana-1);
        output_theta(index,:) = processThetaParams(window, JSONinfo);
%         output_alpha(index,:) = processAlphaParams(window, JSONinfo);
        index=index+1;
    end
end