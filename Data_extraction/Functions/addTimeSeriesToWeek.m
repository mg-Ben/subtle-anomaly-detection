function [time_series_dict] = addTimeSeriesToWeek(TimeSeries, domain, bitsPaquetes, time_series_dict)
    %Function to synchronize time series to a common week-domain from [Lunes
    %00:00:01 -> Lunes 00:00:00 of next week], both included
    
    %Example:--------------------------------------------------------------
    %if time series data exists
    %from [Tuesday at 13:34:23]
    %to   [Friday 09:00:01] (both included), then:
    %this function takes initial time of time serie (Tuesday 13:34:23) and
    %finds the index where from which it has to store time serie vector:
    
    %1 = [Lunes 00:00:01]
    %2 = [Lunes 00:00:02]
    %3 = [Lunes 00:00:03]
    %...
    %60 = [Lunes 00:01:00]
    %...
    %3600 = [Lunes 01:00:00]
    %...
    %24*3600 = [Martes 00:00:00]
    %...
    %2*24*3600 = [Miércoles 00:00:00]
    
    %The index is the number of seconds:
    %result = (weekDay - 1)*24*60*60 + hour*60*60 + minutes*60 + seconds if
    %weekDay indexes from 1 included, or:
    %result = weekDay*24*60*60 + hour*60*60 + minutes*60 + seconds if
    %weekDay indexes from 0 included
    %----------------------------------------------------------------------
    %Maybe the time serie exists from Friday [HH:MM:SS] to Monday
    %[HH:MM:SS] (this is, there exists a time wrapping).
    %In that case, a wrapping must be
    %made to store correctly the time series data.
    %For example, if time series data exists from Friday [14:00:07] to
    %Monday [23:40:22], the first part of time series (Friday [14:00:07]
    %until the end of that week, Sunday [23:59:59]) must be considered a
    %week apart from the remaining part (start of the week Lunes [00:00:00]
    %to Lunes [23:40:22]) so time series must be splitted
    %The wrapping can happen N times (for example, from some Monday to
    %Tuesday of the next week of the next week).
    
    %Take initial epoch timestamp of this time serie from which data exists and convert it to string datetime information:
    date_from = datestr(datetime(TimeSeries(1,1)+7200, 'convertfrom', 'posixtime'));
    date_to = datestr(datetime(TimeSeries(end,1)+7200, 'convertfrom', 'posixtime'));
    index_sec_from = getWeekSeconds(date_from);
    index_sec_to = getWeekSeconds(date_to);
    %Get wrapping limit (limit number in the wrap):
    wrappingLimit = length(domain);
    time_serie_token_cell = getWrappingIndexes(index_sec_from, TimeSeries, wrappingLimit, bitsPaquetes);
    %Add tokens to dictionary:
    time_series_dict = storeTokensInDictionary(time_serie_token_cell, time_series_dict, domain);
    %Maybe the time serie wraps to a week of another file time serie, so
    %it is necessary to save the time serie in a certain row that can be
    %obtained as the week of the year:
    %aggregate_matrix_row_index = weeknum();
    
    %Allocate array in aggregate Matrix:
%     agregado = [agregado; zeros(1, size(agregado, 2))];
%     %Time wrapping check:
%     if index_sec_to < index_sec_from
%         %Time series split:
%         points_to_take = size(agregado,2)-index_sec_from+1;
%         agregado(end, index_sec_from:end) = TimeSeries(1:points_to_take, bitsPaquetes)';
%         agregado = [agregado; zeros(1, size(agregado, 2))];
%         %Take remaining part:
%         try
%             agregado(end, 1:index_sec_to) = TimeSeries(points_to_take+1:end, bitsPaquetes)';
%         catch
%             error
%         end
%     else
%         agregado(end, index_sec_from:index_sec_to) = TimeSeries(:,bitsPaquetes)';
%     end
end