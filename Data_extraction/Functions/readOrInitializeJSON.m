function [jsonObject] = readOrInitializeJSON(JSONoutput_filename, Tventana, n, Granularidad_deteccion, bitsPaquetes, agregado, labels, domainFIT)
    try
        stringJson = fileread(strcat("./Data_extraction_output/", JSONoutput_filename));
        jsonObject = jsondecode(stringJson);
        fprintf("A JSON object with name <%s> already exists.\nChoose one option:\n - 0: Use the existing JSON configuration and continue the window processing from the last window (0)\n - 1: Overwrite the existing JSON with the provided configuration and then process all the windows again (1)\n", JSONoutput_filename);
        fprintf("Existing JSON information: "); jsonObject
        option = input("Option: ");
        while option ~= 0 && option ~= 1
            option = input("Not valid option. Please, provide a valid option: ");
        end
        if option == 1
            fprintf("Overwriting existing JSON...\n");
            jsonObject = buildJSONinfo(Tventana, n, Granularidad_deteccion, bitsPaquetes, 0, agregado, labels, [], [], domainFIT);
        end
    catch
        fprintf("JSON output file not found.\nA new JSON object with name <%s> will be created after this Script...\n", JSONoutput_filename);
        jsonObject = buildJSONinfo(Tventana, n, Granularidad_deteccion, bitsPaquetes, 0, agregado, labels, [], [], domainFIT);
    end
end