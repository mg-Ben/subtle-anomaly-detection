function [agregado, labels] = getAggregateNetTrafficMatrix(filenames, bitsPaquetes, domain)
    agregado = zeros(1, length(domain));
    agregado(1,:) = domain;
    time_series_dict = containers.Map;
    for i=1:length(filenames)
        filename = filenames(i);
        time_serie = load(filename);
        time_series_dict = addTimeSeriesToWeek(time_serie, domain, bitsPaquetes, time_series_dict);
    end
    %From dictionary, get the aggregate Matrix:
    keysArray = keys(time_series_dict);
    labels = [];
    valuesArray = values(time_series_dict);
    for i=1:length(keys(time_series_dict))
        time_serie = valuesArray{i};
        labels = [labels, string(keysArray{i})];
        agregado = [agregado; time_serie];
    end
    agregado(find(agregado <= 0)) = NaN;
end