function [] = writeJSON(filenameOutput, JSON)
    fid = fopen(filenameOutput,'w');
    encodedJSON = jsonencode(JSON);
    try
        fprintf(fid, encodedJSON);
    catch
        fprintf("No se encontró ningún fichero con nombre <%s>. Por favor, asegúrese de crear un fichero inicial vacío con dicho nombre antes de ejecutar este código.\n", filenameOutput);
    end
    fclose(fid); 
end