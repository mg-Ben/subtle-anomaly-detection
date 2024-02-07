function [result] = getWrappingIndexes(index_sec_from, TimeSerie, wrapping_limit, bitsPaquetes)
    %Function to wrap a domain vector to some wrapping limit. For example:
    %[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] to wrapping_limit = 4:
    %[0, 1, 2, 3, 0, 1, 2, 3, 0, 1,  2,  3,  0]
    %The function returns each wrapped token as a cell:
    %[0, 1, 2, 3]
    %[0, 1, 2, 3]
    %[0, 1, 2, 3]
    %[0]
    %It also returns the tokens in time series data. Example:
    %If time series data is:
    %[A, B, C, D, E, F, G, H, I, J, K,  L,   M]
    %And domain vector is:
    %[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    %If domain vector is wrapped as below:
    %[0, 1, 2, 3] --- [A, B, C, D]
    %[0, 1, 2, 3] --- [E, F, G, H]
    %[0, 1, 2, 3] --- [I, J, K, L]
    %[0]          --- [M]
    %The result cell also contains the week number of the year in order to
    %store correctly in aggregate data:
    %[0, 1, 2, 3] --- [A, B, C, D] --- week 45
    %[0, 1, 2, 3] --- [E, F, G, H] --- week 46
    %[0, 1, 2, 3] --- [I, J, K, L] --- week 47
    %[0]          --- [M]          --- week 48
    %In this case, the function receives the index_sec_from and the
    %TimeSerie to get the domain to wrap.
    
    domain_to_wrap = [index_sec_from:index_sec_from+length(TimeSerie(:,1))-1];
    domain_to_wrap_from_zero = domain_to_wrap-1;
    wrapping_times = floor(domain_to_wrap./wrapping_limit);
    wrapped_domain = domain_to_wrap - wrapping_times.*wrapping_limit;
    wrapped_domain_from_one = wrapped_domain+1;
    %How many times the domain was wrapped:
    wrapping_times_unique = unique(wrapping_times);
    result = cell(length(wrapping_times_unique), 3);
    token_from = 1;
    for i=0:length(wrapping_times_unique)-1
        domain_index_access = find(wrapping_times == i);
        token_to = token_from + length(domain_index_access)-1;
        result{i+1,1} = wrapped_domain_from_one(domain_index_access);
        result{i+1,2} = TimeSerie(token_from:token_to, bitsPaquetes)';
        %Get week of the year:
        %Take one day (anyone) of TimeSerie token: for example, the first
        %one:
        date = datestr(datetime(TimeSerie(token_from, 1)+7200, 'convertfrom', 'posixtime'));
        result{i+1,3} = weeknum(date);
        token_from = token_to+1;
    end
end