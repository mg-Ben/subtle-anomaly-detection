%-------------------------------------------------------------------------
% M20230722_TrendDynamics : Obtener coeficientes de regresión polinómica 
%                           para ventanas de Tventana minutos que se 
%                           deslizan cada segundo:
%                           Autor: Benjamín Martín 
%                           Revisión: Luis de Pedro 
%--------------------------------------------------------------------------
%Obtener parámetros alpha-stable y coeficientes de regresión polinómica
%para ventanas de Tventana minutos que se deslizan cada segundo:
clear all; close all; clc; warning off
addpath('./Functions')

%PARAMETROS DE ENTRADA:----------------------------------------------------
WindowsToSimulate = 400000; %Cuantas ventanas se desean simular

JSONoutput_filename = "trendDynamicsOutputTEST_n4_T30.json"; %Nombre del fichero JSON de salida. Ejemplo: outputJSON.json
Tventana = 30; %[min] (Tamaño de ventana deslizante T)
n = 4; %Grado de la regresión polinómica
Granularidad_deteccion = 180; %= scope del sistema [s] (alcance o tiempo de incertidumbre de predicción)
bitsPaquetes = 3; %Indica si trabajar con bits/s (2) o packets/s (3)
filenames = ["./Output_files/<yourfile1.csv>";
             "./Output_files/<yourfile2.csv>"];
%--------------------------------------------------------------------------
%Obtener la matriz con todas las series temporales de cada semana
%NOTA: Dado que todas las series se ordenan semanalmente, algunas
%no tienen datos para ciertos días de la semana (por ejemplo, tal vez la
%semana 'X' del mes 'Y' no tenga datos para el Lunes y el Martes). Por esta
%razón, es posible ver valores NaN al comienzo en las bases de datos de los
%parámetros theta y alpha.
%LA PRIMERA FILA DE LA MATRIZ DE AGREGADO ES EL DOMINIO:
domain = 1:7*24*60*60; %[1 = Lunes 00:00:01 -> 7*24*60*60 = Lunes (semana siguiente) 00:00:00]
fprintf("Reading time series data...\n");
[agregado, labels] = getAggregateNetTrafficMatrix(filenames, bitsPaquetes, domain);
labels = ["domain", labels]';
fprintf("Time series data read finished\n");

%Obtener series temporales con las dinámicas de la tendencia polinómica y los parámetros alpha-estable:
%Definir el dominio de la regresión:
Tsventana = Tventana*60;
domainFIT = getDomainFIT(Tsventana, Granularidad_deteccion);

%Antes de comenzar a procesar las series temporales semanales, se debe
%comprobar si ya existe un fichero JSON con la informacion del procesado:
JSONinfo = readOrInitializeJSON(JSONoutput_filename, Tventana, n, Granularidad_deteccion, bitsPaquetes, agregado, labels, domainFIT);
JSONinfo = processTrendDynamics(WindowsToSimulate, JSONinfo);

%Exportar las dinámicas de la tendencia y los parámetros alpha-stable:
fprintf("\nProcessing finished. JSON data:\n"); JSONinfo
writeJSON(strcat("./Data_extraction_output/", JSONoutput_filename), JSONinfo);

%Representación:
% fprintf("Plotting data...\n");
% key_TP = strcat(strcat(strcat("TP", string(JSONinfo.Tsventana/60)), '_'), string(JSONinfo.n));
% key_AP = strcat(strcat(strcat("AP", string(JSONinfo.Tsventana/60)), '_'), string(JSONinfo.n));
% thetas = JSONinfo.(key_TP);
% alphas = JSONinfo.(key_AP);
% for c=1:n+1 %Por cada coeficiente:
%     figure;
%     for i=1:size(thetas,2)/(n+1) %Sacaremos size(thetas,2)/(n+1) gráficas, una por cada serie temporal
%         %Por cada serie temporal:
%         thetas_window = thetas(:, (i-1)*(n+1)+c);
%         %Evolución temporal de los parámetros theta:
%         plot([1:JSONinfo.Number_of_simulated_windows], thetas_window); axis tight; grid on; title('Theta(#ventana)'); xlabel('#ventana'); ylabel(strcat('Theta_', string(c-1)));
%         hold on;
%     end
%     hold off;
% end
% 
% for a=1:4 %Por cada parámetro alpha-stable posible (solo hay 4)
%     figure;
%     for i=1:size(alphas,2)/4 %Por cada serie temporal
%         alphas_window = alphas(:,(i-1)*4+a);
%         plot([1:JSONinfo.Number_of_simulated_windows], alphas_window); axis tight; grid on; title('Alphas(#ventana)'); xlabel('#ventana');
%         if(a == 1)
%             ylabel('Alpha');
%         end
%         if(a == 2)
%             ylabel('Beta');
%         end
%         if(a == 3)
%             ylabel('Gamma');
%         end
%         if(a == 4)
%             ylabel('Delta');
%         end
%         hold on;
%     end
%     hold off;
% end