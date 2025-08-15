for i in $(seq -w 0 $1); 
do sllm-cli deploy --config "synth_model_$i.json";
done
