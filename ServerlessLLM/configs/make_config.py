config_template = \
'''
{
    "model": "$1",
    "backend": "vllm",
    "num_gpus": 1,
    "auto_scaling_config": {
        "metric": "concurrency",
        "target": 1,
        "min_instances": 0,
        "max_instances": 1,
        "keep_alive": 0
    },
    "backend_config": {
        "pretrained_model_name_or_path": "/root/models/$2",
        "device_map": "auto",
        "torch_dtype": "float16",
        "hf_model_class": "AutoModelForCausalLM",
        "enable_lora": false,
	    "enforce_eager": true,
	    "enable_prefix_caching": false
    }
}
'''

for i in range(40):
    with open(f'synth_model_{i:02d}.json', 'w') as f:
        model_path = [
            "Qwen2.5-7B-Instruct",
            "Qwen2.5-7B-Instruct",
            "Qwen2.5-7B-Instruct",
            "Yi-1.5-9B-Chat",
            "Llama-2-13b-chat-ms",
        ][i % 5]
        f.write(config_template.replace('$1', f'synth_model_{i:02d}').replace('$2', model_path))

