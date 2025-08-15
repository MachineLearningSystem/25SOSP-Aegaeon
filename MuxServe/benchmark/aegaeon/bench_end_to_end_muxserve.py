import os
import json
import yaml
import argparse

from pprint import pprint
from typing import Dict, List

END_TO_END_DIR = os.path.dirname(__file__)
PROJ_DIR = f"{os.path.dirname(__file__)}/../.."

MODEL_TO_PATH = {
    "llama-7b": "/root/models/llama-7b",
    "llama-13b": "/root/models/llama-13b",
}
# path to `ShareGPT_V3_unfiltered_cleaned_split.json`
SHAREGPT_PATH = "/root/ShareGPT_V3_unfiltered_cleaned_split.json"
# statistic cost file
COST_FILE = f"{PROJ_DIR}/examples/placement/llama.json"


def gen_models_yaml(
    nnodes: int,
    ngpus_per_node: int,
    model_to_rate: Dict[str, List[float]],
    dump_path: str,
):
    data = {
        "cluster": {
            "nnodes": nnodes,
            "ngpus_per_node": ngpus_per_node
        },
        "models": [],
    }

    model_id = 0
    for model, rates in model_to_rate.items():
        for rate in rates:
            data["models"].append({
                "name": f"llm-{model_id}",
                "model": MODEL_TO_PATH[model],
                "rate": rate
            })
            model_id += 1

    with open(dump_path, "w") as fp:
        yaml.dump(data, fp, sort_keys=False)


def get_workload_from_optimized_placement(
    info: Dict[str, dict],
    time: int,
    models_yaml: str,
    dump_dir: str,
    **kwargs,
):
    from muxserve.muxsched.workload_utils import get_workloads_info_from_yaml, generate_workload, sample_request_datas

    workload_infos = get_workloads_info_from_yaml(models_yaml)

    llm_tpt = []
    info.pop("muxserve_tpt")
    for mesh_id, llms in info.items():
        for llm_id, llm_info in llms.items():
            llm_tpt.append((llm_id, llm_info["expected_tpt"]))

    llm_tpt.sort()
    pprint(f"llm_tpt:\n{llm_tpt}")

    sampled_req = []
    num_req = []
    for llm_id, model_tpt in llm_tpt:
        cur_num_req = int(model_tpt * time * 1.1)
        num_req.append(cur_num_req)
        sampled_req.append(
            sample_request_datas(cur_num_req,
                                 SHAREGPT_PATH,
                                 tokenized_cache_path=None))
    max_num_req = max(num_req)

    kwargs.update({
        "sampled_requests": sampled_req,
        "num_requests": num_req,
    })
    output_file = os.path.join(
        dump_dir, f"sharegpt_n{len(llm_tpt)}_req.json")
    
    generate_workload(workload_infos, output_file, **kwargs)


def get_placement_from_cfg(models_yaml: str,
                           costfile: str,
                           is_greedy=False,
                           dump_to_yaml=True,
                           dump_dir: str = None,
                           verbose: bool = False):
    from muxserve.muxsched.placement import PlacementOptimizer

    opt = PlacementOptimizer(models_yaml, costfile)

    return opt.optimize(is_greedy,
                        dump_dir=dump_dir,
                        dump_to_yaml=dump_to_yaml,
                        verbose=verbose)


def gen_config(
    config_dir: str, 
    workloads_dir: str,
    num_models: List[int],
    arrival_rate: List[float],
    inlen_scale: float = 1.0,
    outlen_scale: float = 1.0,
):
    nnodes = 1
    ngpus_per_node = 8

    tmp_cfg = f"/tmp/tmp_model_cfg.yaml"

    if not os.path.exists(config_dir):
        os.makedirs(config_dir, exist_ok=True)

    flog = open(f"{config_dir}/gen_pl.log", "w")

    for rate in arrival_rate:
        for models in num_models:
            model2num = {
                "llama-7b": 0,
                "llama-13b": 0,
            }
            for i in range(models):
                if i % 5 == 4:
                    model2num["llama-13b"] += 1
                else:
                    model2num["llama-7b"] += 1

            cfg_dir = f"{config_dir}/models{models}_rate{rate}_scale_{inlen_scale}_{outlen_scale}"
            if not os.path.exists(cfg_dir):
                os.makedirs(cfg_dir, exist_ok=True)

            rate_map = {
                k: [rate for _ in range(v)] for k, v in model2num.items()
            }
            gen_models_yaml(nnodes, ngpus_per_node, rate_map, tmp_cfg)

            muxserve_placement = get_placement_from_cfg(tmp_cfg,
                                                COST_FILE,
                                                dump_to_yaml=True,
                                                dump_dir=cfg_dir,
                                                verbose=False)
            
            if muxserve_placement is None:
                # Retry with all 7B models
                model2num = {"llama-7b": models}
                rate_map = {
                    k: [rate for _ in range(v)] for k, v in model2num.items()
                }
                gen_models_yaml(nnodes, ngpus_per_node, rate_map, tmp_cfg)
                muxserve_placement = get_placement_from_cfg(tmp_cfg,
                                                            COST_FILE,
                                                            dump_to_yaml=True,
                                                            dump_dir=cfg_dir,
                                                            verbose=False)
            
            if muxserve_placement is None:
                print(f"MuxServe cannot find a placement for {models} models and rate = {rate}; skipping...")
                continue
                                                
            workloads_dump_dir = f"{workloads_dir}/models{models}_rate{rate}_scale_{inlen_scale}_{outlen_scale}"
            if not os.path.exists(workloads_dump_dir):
                os.makedirs(workloads_dump_dir, exist_ok=True)

            workload_args = {
                "start": 0,
                "duration": 60,
                "distribution": "poisson",
                "prompt_distribution": None,
                "use_share_gpt": True,
                "prompt_len": None,
                "output_len": None,
                "dataset": SHAREGPT_PATH,
            }
            get_workload_from_optimized_placement(
                muxserve_placement,
                time=60,
                models_yaml=tmp_cfg,
                dump_dir=workloads_dump_dir,
                **workload_args)

            flog.write(f"{cfg_dir}\n{json.dumps(muxserve_placement)}\n")

    flog.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-models", type=int, nargs="+")
    parser.add_argument("--arrival-rate", type=float, nargs="+")
    parser.add_argument("--inlen-scale", type=float, default=1.0)
    parser.add_argument("--outlen-scale", type=float, default=1.0)
    args = parser.parse_args()

    muxserve_cfg_dir = f"{END_TO_END_DIR}/model_cfgs"
    workloads_dir = f"{END_TO_END_DIR}/workloads"
    models_yaml_path = f"{END_TO_END_DIR}/models.yaml"

    gen_config(muxserve_cfg_dir, workloads_dir, args.num_models, args.arrival_rate, inlen_scale=args.inlen_scale, outlen_scale=args.outlen_scale)

