import os
import json
import time
import datetime
from tqdm import tqdm
import numpy as np
import random
import argparse
from tiny_tools.read_json import read_config
import torch.multiprocessing as mp
from torch.multiprocessing import Process
import torch
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, GenerationConfig
import torch.distributed as dist
import infer as infer_module
from infer import get_any_chat_api
from tiny_tools.tensor_tools import idx_tracer


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None,
                        choices=["llama2-7b-chat-4k", "longchat-v1.5-7b-32k", "xgen-7b-8k", "internlm-7b-8k",
                                 "vicuna-v1.5-7b-16k",
                                 "my_model"])
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument("--cfg", default=None)
    parser.add_argument("--ws", default=2, type=int, help='world size')
    parser.add_argument("--task_start_id", default=0, type=int)
    parser.add_argument("--task", default=None, type=str)
    parser.add_argument("--n", default=0, type=int, help="Number of samples to use (0 for all)")
    return parser.parse_args(args)


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name or 'llama-2' in model_name.lower():
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt


def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


def get_pred(rank, world_size, data, max_length, max_gen, prompt_format, dataset, device, model_name, model2path,
             out_path, args):
    seed_everything(42)
    d_cfg = read_config(args.cfg)
    device = 0
    # x = torch.rand(100_000, 100_000, device=0)
    # device_num = torch.cuda.device_count()
    # if d_cfg.get('use_multi_gpus', False):
    #     device = rank % device_num
    #     os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    #     device = 0
    #     assert torch.cuda.device_count() == 1

    model, tokenizer, model_max_length = load_model_and_tokenizer(model2path[model_name], model_name, device, args.cfg)
    if model_max_length is not None:
        max_length = model_max_length
        print(f"max_length is set to {max_length}")

    for json_obj in tqdm(data, desc=f'{dataset}'):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle,
        # since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = (tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) +
                      tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True))

        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc",
                           "repobench-p"]:  # chat models are better off without build prompts on these tasks
            if 'my_model' not in model_name:
                prompt = build_chat(tokenizer, prompt, model_name)
            else:
                prompt = build_chat(tokenizer, prompt, d_cfg['model_name'])

        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        if dataset == "samsum":
            # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            if "my_model" not in model_name:
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    min_length=context_length + 1,
                    eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                )[0]
            else:
                output = model(
                    prompt, generation_config=None,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    min_length=context_length + 1,
                    eos_token_id=[tokenizer.eos_token_id,
                                  tokenizer.encode("\n", add_special_tokens=False)[-1]],
                    skip_special_tokens=True)
        else:
            if "my_model" not in model_name:
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                )[0]
            else:
                output = model(
                    prompt, generation_config=None,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    skip_special_tokens=True
                )
        if not isinstance(output, str):
            pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        else:
            pred = output

        pred = post_process(pred, model_name)
        # Collect per-sample inference metadata
        meta = infer_module.last_inference_meta.copy()
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"],
                       "length": json_obj["length"],
                       "input_len": meta.get("input_len"),
                       "output_len": meta.get("output_len"),
                       "num_selected_kv": meta.get("num_selected_kv")}, f, ensure_ascii=False)
            f.write('\n')

        # 用来分析模型性质的code:::
        if os.environ.get('SAVE_SELECTED_IDX', False):
            idx_tracer.save_idx()
            # 提前终止
            if idx_tracer.num_samples > 20:
                return


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(path, model_name, device, cfg):
    max_length = None
    if "chatglm" in model_name or "internlm" in model_name or "xgen" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(
            device)
    if 'my_model' in model_name:
        model, tokenizer, max_length, other_kwargs = get_any_chat_api(cfg)
        tokenizer.eos_token_id = other_kwargs['eos_token_id']
        print("EOS is", tokenizer.eos_token_id)
    elif "llama2" in model_name:
        # replace_llama_attn_with_flash_attn()
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                 attn_implementation="flash_attention_2").to(device)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import load_model
        replace_llama_attn_with_flash_attn()
        model, _ = load_model(
            path,
            device='cpu',
            num_gpus=0,
            load_8bit=False,
            cpu_offloading=False,
            debug=False,
        )
        model = model.to(device)
        model = model.bfloat16()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    # model = model.eval_dir()
    return model, tokenizer, max_length


def load_dataset(path, mode='r'):
    data = [json.loads(line) for line in open(path, mode, encoding="utf-8")]
    return data


class CustomProcess(Process):
    def __init__(self, env_var_key, env_var_value, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env_var_key = env_var_key
        self.env_var_value = env_var_value
        os.environ[self.env_var_key] = self.env_var_value

    def run(self):
        # 在子进程中设置环境变量
        os.environ[self.env_var_key] = self.env_var_value
        # 调用原始的进程执行目标函数
        super().run()


if __name__ == '__main__':
    args = parse_args()
    world_size = args.ws
    mp.set_start_method('spawn', force=True)

    model2path = json.load(open("benchmark/long_bench/config/model2path.json", "r"))
    model2maxlen = json.load(open("benchmark/long_bench/config/model2maxlen.json", "r"))
    model_name = args.model
    # define your model
    max_length = model2maxlen.get(model_name, -1)
    d_cfg = read_config(args.cfg)
    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news",
                    "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = [
            "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique",
            "dureader", "gov_report", "qmsum",
            # "multi_news", # 找不到
            "vcsum", "trec", "triviaqa",
            # "samsum", # 找不到
            "lsht",
            "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"
        ]
    datasets = datasets[args.task_start_id:]
    if args.task is not None:
        datasets = args.task.split(',')
        # print("for debug", datasets)
    # we design specific prompt format and max generation length for each task,
    # feel free to modify them to optimize model output
    dataset2prompt = json.load(open("benchmark/long_bench/config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("benchmark/long_bench/config/dataset2maxlen.json", "r"))

    # Increase max_gen when thinking mode is enabled
    enable_thinking = d_cfg.get("enable_thinking", False)
    thinking_extra_tokens = d_cfg.get("thinking_extra_tokens", 4096)
    if enable_thinking:
        print(f"[Thinking ON] Adding {thinking_extra_tokens} extra tokens to max_gen for each dataset")
        dataset2maxlen = {k: v + thinking_extra_tokens for k, v in dataset2maxlen.items()}

    # Track experiment timing and memory
    exp_start_time = time.time()
    gpu_mem_before = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    per_dataset_stats = {}

    base_path = ''
    if os.environ.get("NO_NAS", False):
        base_path = '/jitai/'
    for dataset in datasets:
        if args.e:
            data = load_dataset(f'benchmark/long_bench/data/{dataset}_e.jsonl', 'r')
            if not os.path.exists(f"{base_path}benchmark/long_bench/pred_e/{model_name}/{args.cfg}"):
                os.makedirs(f"{base_path}benchmark/long_bench/pred_e/{model_name}/{args.cfg}", exist_ok=True)
            out_path = f"{base_path}benchmark/long_bench/pred_e/{model_name}/{args.cfg}/{dataset}.jsonl"
        else:
            data = load_dataset(f'benchmark/long_bench/data/{dataset}.jsonl', 'r')
            if not os.path.exists(f"{base_path}benchmark/long_bench/pred/{model_name}/{args.cfg}"):
                os.makedirs(f"{base_path}benchmark/long_bench/pred/{model_name}/{args.cfg}", exist_ok=True)
            out_path = f"{base_path}benchmark/long_bench/pred/{model_name}/{args.cfg}/{dataset}.jsonl"
        with open(out_path, 'w') as _in:
            pass  # 清空里面的内容
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        if args.n > 0:
            data_all = data_all[:args.n]
        # data_subsets = [data_all[i::world_size] for i in range(world_size)]
        processes = []
        torch.cuda.empty_cache()
        ds_start = time.time()
        get_pred(0, world_size, data_all, max_length,
                 max_gen, prompt_format, dataset, None, model_name, model2path,
                 out_path, args)
        ds_elapsed = time.time() - ds_start
        ds_peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0
        per_dataset_stats[dataset] = {
            "num_samples": len(data_all),
            "elapsed_time_s": round(ds_elapsed, 2),
            "peak_gpu_memory_gb": round(ds_peak_mem, 2),
        }
    # ===== Run evaluation and generate experiment log =====
    exp_total_time = time.time() - exp_start_time
    peak_gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0

    # Run evaluation (inline, same logic as eval.py)
    from metrics import (
        qa_f1_score, rouge_zh_score, qa_f1_zh_score, rouge_score,
        classification_score, retrieval_score, retrieval_zh_score,
        count_score, code_sim_score,
    )
    _dataset2metric = {
        "narrativeqa": qa_f1_score, "qasper": qa_f1_score,
        "multifieldqa_en": qa_f1_score, "multifieldqa_zh": qa_f1_zh_score,
        "hotpotqa": qa_f1_score, "2wikimqa": qa_f1_score, "musique": qa_f1_score,
        "dureader": rouge_zh_score, "gov_report": rouge_score, "qmsum": rouge_score,
        "multi_news": rouge_score, "vcsum": rouge_zh_score,
        "trec": classification_score, "triviaqa": qa_f1_score, "samsum": rouge_score,
        "lsht": classification_score, "passage_retrieval_en": retrieval_score,
        "passage_count": count_score, "passage_retrieval_zh": retrieval_zh_score,
        "lcc": code_sim_score, "repobench-p": code_sim_score,
    }

    if args.e:
        pred_dir = f"{base_path}benchmark/long_bench/pred_e/{model_name}/{args.cfg}"
    else:
        pred_dir = f"{base_path}benchmark/long_bench/pred/{model_name}/{args.cfg}"

    scores = {}
    sample_details = {}  # per-dataset list of {input_len, output_len, num_selected_kv}
    for filename in os.listdir(pred_dir):
        if not filename.endswith("jsonl"):
            continue
        ds_name = filename.split('.')[0]
        predictions, answers = [], []
        ds_details = []
        with open(f"{pred_dir}/{filename}", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                all_classes = data["all_classes"]
                ds_details.append({
                    "input_len": data.get("input_len"),
                    "output_len": data.get("output_len"),
                    "num_selected_kv": data.get("num_selected_kv"),
                })
        sample_details[ds_name] = ds_details
        if not predictions:
            continue
        try:
            total_score = 0.
            for pred_text, ground_truths in zip(predictions, answers):
                s = 0.
                if ds_name in ["trec", "triviaqa", "samsum", "lsht"]:
                    pred_text = pred_text.lstrip('\n').split('\n')[0]
                for gt in ground_truths:
                    s = max(s, _dataset2metric[ds_name](pred_text, gt, all_classes=all_classes))
                total_score += s
            scores[ds_name] = round(100 * total_score / len(predictions), 2)
        except Exception as e:
            print(f"eval error in {ds_name}: {e}")

    # Save result.json
    result_path = f"{pred_dir}/result.json"
    with open(result_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
    print(f"Scores: {scores}")

    # Compute KV cache ratio
    _do_sel = [int(i) for i in d_cfg.get("do_select_layers", "0").split(",")]
    _nwait = d_cfg.get("num_wait_load_layers", 2)
    _token_ratio = d_cfg.get("num_of_selected_tokens", 4096)
    _num_layers = None
    # Try to get num_hidden_layers from model config
    model_cfg_path = os.path.join(d_cfg["model_name"], "config.json")
    if os.path.exists(model_cfg_path):
        with open(model_cfg_path) as f:
            _num_layers = json.load(f).get("num_hidden_layers")
    kv_cache_info = {}
    if _num_layers and isinstance(_token_ratio, float):
        _full_attn = list(range(0, _do_sel[0])) + _do_sel + [_num_layers]
        _nf, _ns = 0, 0
        for _l in _full_attn:
            _r = _num_layers
            for _i in range(_l + 1, _num_layers + 1):
                if _i in _full_attn:
                    _r = _i
                    break
            _nf += min(_r, _l + _nwait + 1) - _l
            _ns += max(0, _r - min(_r, _l + _nwait + 1))
        overall_ratio = (_nf * 1.0 + _ns * _token_ratio) / _num_layers
        kv_cache_info = {
            "per_layer_token_ratio": _token_ratio,
            "full_cache_layers": _nf,
            "sparse_cache_layers": _ns,
            "total_layers": _num_layers,
            "overall_kv_cache_ratio": round(overall_ratio, 4),
        }

    # GPU info
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"

    # Summarize per-dataset sample details (input_len, output_len, num_selected_kv)
    inference_stats = {}
    for ds_name, details in sample_details.items():
        input_lens = [d["input_len"] for d in details if d.get("input_len") is not None]
        output_lens = [d["output_len"] for d in details if d.get("output_len") is not None]
        selected_kvs = [d["num_selected_kv"] for d in details if d.get("num_selected_kv") is not None]
        stat = {}
        if input_lens:
            stat["input_len"] = {"mean": round(np.mean(input_lens), 1), "min": min(input_lens), "max": max(input_lens)}
        if output_lens:
            stat["output_len"] = {"mean": round(np.mean(output_lens), 1), "min": min(output_lens), "max": max(output_lens)}
        if selected_kvs:
            stat["num_selected_kv"] = {"mean": round(np.mean(selected_kvs), 1), "min": min(selected_kvs), "max": max(selected_kvs)}
        if input_lens and selected_kvs:
            ratios = [s / i for s, i in zip(selected_kvs, input_lens) if i > 0]
            stat["actual_kv_select_ratio"] = {"mean": round(np.mean(ratios), 4), "min": round(min(ratios), 4), "max": round(max(ratios), 4)}
        stat["samples"] = details
        inference_stats[ds_name] = stat

    # Build experiment log
    exp_log = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config_path": args.cfg,
        "model_name": d_cfg.get("model_name", "unknown"),
        "model_cls": d_cfg.get("model_cls", "unknown"),
        "enable_thinking": enable_thinking,
        "num_samples_per_dataset": args.n if args.n > 0 else "all",
        "max_context_len": d_cfg.get("max_context_len", -1),
        "kv_cache": kv_cache_info,
        "scores": scores,
        "avg_score": round(np.mean(list(scores.values())), 2) if scores else 0,
        "per_dataset_stats": per_dataset_stats,
        "inference_stats": inference_stats,
        "total_elapsed_time_s": round(exp_total_time, 2),
        "peak_gpu_memory_gb": round(peak_gpu_mem, 2),
        "gpu": gpu_name,
        "datasets_evaluated": list(scores.keys()),
    }

    # Save log file
    log_dir = f"{pred_dir}"
    log_filename = f"experiment_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_path = os.path.join(log_dir, log_filename)
    with open(log_path, "w") as f:
        json.dump(exp_log, f, ensure_ascii=False, indent=4)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Experiment Summary")
    print(f"{'='*60}")
    print(f"  Model:            {d_cfg.get('model_name', 'unknown')}")
    print(f"  Thinking:         {'ON' if enable_thinking else 'OFF'}")
    print(f"  Datasets:         {list(scores.keys())}")
    print(f"  Avg Score:        {exp_log['avg_score']}")
    print(f"  Total Time:       {exp_total_time:.1f}s")
    print(f"  Peak GPU Memory:  {peak_gpu_mem:.2f} GB")
    print(f"  GPU:              {gpu_name}")
    if kv_cache_info:
        print(f"  KV Cache Ratio:   {kv_cache_info['overall_kv_cache_ratio']*100:.2f}% "
              f"(per-layer: {kv_cache_info['per_layer_token_ratio']*100:.2f}%, "
              f"full: {kv_cache_info['full_cache_layers']}, sparse: {kv_cache_info['sparse_cache_layers']})")
    print(f"  Scores:           {scores}")
    for ds_name, stat in inference_stats.items():
        if "input_len" in stat and "num_selected_kv" in stat:
            print(f"  [{ds_name}] input_len: {stat['input_len']}, "
                  f"output_len: {stat.get('output_len', 'N/A')}, "
                  f"selected_kv: {stat['num_selected_kv']}, "
                  f"actual_ratio: {stat.get('actual_kv_select_ratio', 'N/A')}")
    print(f"  Log saved to:     {log_path}")
    print(f"{'='*60}")
