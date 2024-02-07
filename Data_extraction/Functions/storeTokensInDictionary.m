function [dictionary] = storeTokensInDictionary(time_serie_tokens, dictionary, domain)
    for i=1:size(time_serie_tokens, 1)
        key_to_store = num2str(time_serie_tokens{i,3});
        data_vector = zeros(1, length(domain));
        indexes = time_serie_tokens{i,1};
        time_series_data = time_serie_tokens{i,2};
        data_vector(indexes) = time_series_data; %Vector to store
        if(isKey(dictionary, key_to_store))
            %Retrieve existing data and sum vectors:
            dictionary(key_to_store) = dictionary(key_to_store)+data_vector;
        else
            dictionary(key_to_store) = data_vector;
        end
    end
end