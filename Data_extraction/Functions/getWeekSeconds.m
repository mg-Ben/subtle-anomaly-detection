function [week_seconds] = getWeekSeconds(stringDate)
    %Obtain the weekDay number:
    [weekDay_numb, weekDay_char] = weekday(stringDate); %1: domingo, 2: lunes, 3: martes..., 7: sábado
    %Convert to this format: 0: lunes, 1: martes..., 5: sábado, 6: domingo
    weekDay_numb = weekDay_numb-2;
    if(weekDay_numb < 0)
        weekDay_numb = 6; %Domingo
    end
    %Get the houminsec string:
    [day, hourminsec] = strtok(stringDate);
    %Get the hours:
    [hour, minsec] = strtok(hourminsec, ':');
    hour = str2double(hour(2:end));
    %Get the mins:
    [min, sec] = strtok(minsec, ':');
    min = str2double(min);
    %Get seconds:
    sec = str2double(sec(2:end));
    %Result:
    week_seconds = weekDay_numb*24*60*60 + hour*60*60 + min*60 + sec;
end