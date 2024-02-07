#!/bin/bash

ls Input_files | while read -r filename_input; do
	echo "-----------------------------------------------------------------------------------------------------"
	filename_input="./Input_files/"$filename_input
	filename_output=$(basename "$filename_input")
	filename_output="${filename_output%.*}"

	export LC_NUMERIC="en_US.UTF-8"

	referencia_tiempo=$(awk -F '[-:,]' 'BEGIN {c = 0}{
		if(c != 0){
			if($1 " " $2 " " $3 " " $4 " " $5 != tfecha_was){
				tf = mktime($1 " " $2 " " $3 " " $4 " " $5)
			}
			
			if($6 == 0){
				ti_is = tf - 1
			}
			else{
				ti_is = int(tf - $6)
			}
			
			if(ti_is < ti_first){
				ti_first = ti_is
			}
			tfecha_was = $1 " " $2 " " $3 " " $4 " " $5
		}
		else{
			if($1 > 0){
				tfecha_was = $1 " " $2 " " $3 " " $4 " " $5
				tf = mktime(tfecha_was)
				
				if($6 == 0){
					ti_first = tf - 1
				}
				else{
					ti_first = int(tf - $6)
				}
				c = 1
			}
		}
	}END{print int(sprintf("%d", ti_first))}' $filename_input)

	echo "Time reference obtained:" $referencia_tiempo #Para hacer un print de la variable de referencia_tiempo
	echo "Converting" $filename_input "into time series... (This might take hours)"

	if [ ! -d "output" ]; then
			# Si el directorio no existe, entonces lo crea
			mkdir "Output_files"
	fi

	awk -F '[-:,]' -v tref="$referencia_tiempo" '{
		tf_flujo = mktime($1 " " $2 " " $3 " " $4 " " $5)-tref
		dur_flujo = $6
		
		npaquetes = $15
		nbits = $16*8
		if(dur_flujo >= 0 && dur_flujo <= 1){
			bps[tf_flujo] += nbits
			pps[tf_flujo] += npaquetes
		}
		else{
			ti_flujo_eff = tf_flujo - dur_flujo
			ti_flujo = int(sprintf("%d", ti_flujo_eff))
			for(i = ti_flujo + 1; i <= tf_flujo; i++){
				if(i == ti_flujo + 1){
					bps[i] += (1 - (ti_flujo_eff - ti_flujo))*nbits/dur_flujo
					pps[i] += (1 - (ti_flujo_eff - ti_flujo))*npaquetes/dur_flujo
				}
				else{
					bps[i] += nbits/dur_flujo
					pps[i] += npaquetes/dur_flujo
				}
			}
		}
	}END{for(i=1; i<=length(bps); i++){printf("%d, %.9f, %.9f\n", i+tref, bps[i], pps[i]);}}' $filename_input > "./Output_files/$filename_output"

	echo $filename_input "finished. Output is" "$filename_output"
done