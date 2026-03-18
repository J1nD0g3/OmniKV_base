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
        # Dynamically set max_gen for thinking mode: max_context_len - input_len
        _max_gen = max_gen
        if d_cfg.get("enable_thinking", False):
            _max_gen = max(max_gen, d_cfg.get("max_context_len", 32000) - context_length)
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
                    max_new_tokens=_max_gen,
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
                    max_new_tokens=_max_gen,
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

    enable_thinking = d_cfg.get("enable_thinking", False)
    thinking_max_context = d_cfg.get("max_context_len", 32000)
    if enable_thinking:
        print(f"[Thinking ON] max_gen will be dynamically set to max_context_len({thinking_max_context}) - input_len per sample")

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
    _dataset2metric_name = {
        "narrativeqa": "qa_f1_score", "qasper": "qa_f1_score",
        "multifieldqa_en": "qa_f1_score", "multifieldqa_zh": "qa_f1_zh_score",
        "hotpotqa": "qa_f1_score", "2wikimqa": "qa_f1_score", "musique": "qa_f1_score",
        "dureader": "rouge_zh_score", "gov_report": "rouge_score", "qmsum": "rouge_score",
        "multi_news": "rouge_score", "vcsum": "rouge_zh_score",
        "trec": "classification_score", "triviaqa": "qa_f1_score", "samsum": "rouge_score",
        "lsht": "classification_score", "passage_retrieval_en": "retrieval_score",
        "passage_count": "count_score", "passage_retrieval_zh": "retrieval_zh_score",
        "lcc": "code_sim_score", "repobench-p": "code_sim_score",
    }
    _dataset2group = {
        "narrativeqa": "Single-Document QA", "qasper": "Single-Document QA",
        "multifieldqa_en": "Single-Document QA", "multifieldqa_zh": "Single-Document QA",
        "hotpotqa": "Multi-Document QA", "2wikimqa": "Multi-Document QA",
        "musique": "Multi-Document QA", "dureader": "Multi-Document QA",
        "gov_report": "Summarization", "qmsum": "Summarization",
        "multi_news": "Summarization", "vcsum": "Summarization",
        "trec": "Few-shot Learning", "triviaqa": "Few-shot Learning",
        "samsum": "Few-shot Learning", "lsht": "Few-shot Learning",
        "passage_count": "Synthetic Tasks", "passage_retrieval_en": "Synthetic Tasks",
        "passage_retrieval_zh": "Synthetic Tasks",
        "lcc": "Code Completion", "repobench-p": "Code Completion",
    }

    if args.e:
        pred_dir = f"{base_path}benchmark/long_bench/pred_e/{model_name}/{args.cfg}"
    else:
        pred_dir = f"{base_path}benchmark/long_bench/pred/{model_name}/{args.cfg}"

    scores = {}
    sample_details = {}
    per_sample_scores = {}
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
            sample_scores = []
            for pred_text, ground_truths in zip(predictions, answers):
                s = 0.
                if ds_name in ["trec", "triviaqa", "samsum", "lsht"]:
                    pred_text = pred_text.lstrip('\n').split('\n')[0]
                for gt in ground_truths:
                    s = max(s, _dataset2metric[ds_name](pred_text, gt, all_classes=all_classes))
                sample_scores.append(s)
            scores[ds_name] = round(100 * np.mean(sample_scores), 2)
            stderr = round(100 * np.std(sample_scores) / np.sqrt(len(sample_scores)), 4) if len(sample_scores) > 1 else 0
            per_sample_scores[ds_name] = {"scores": sample_scores, "stderr": stderr}
        except Exception as e:
            print(f"eval error in {ds_name}: {e}")

    # Save result.json
    result_path = f"{pred_dir}/result.json"
    with open(result_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)

    # Compute KV cache ratio
    _do_sel = [int(i) for i in d_cfg.get("do_select_layers", "0").split(",")]
    _nwait = d_cfg.get("num_wait_load_layers", 2)
    _token_ratio = d_cfg.get("num_of_selected_tokens", 4096)
    _num_layers = None
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

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"

    # ===== Create log directory structure =====
    model_short = os.path.basename(d_cfg.get("model_name", "unknown"))
    think_str = "think-on" if enable_thinking else "think-off"
    run_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{model_short}_longbench_{think_str}_{run_timestamp}"
    log_base = "logs"
    run_dir = os.path.join(log_base, run_name)
    samples_dir = os.path.join(run_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    # ===== Save per-sample details per dataset =====
    for ds_name, details in sample_details.items():
        sample_file = os.path.join(samples_dir, f"{ds_name}.json")
        ds_sample_data = []
        ds_scores_list = per_sample_scores.get(ds_name, {}).get("scores", [])
        for i, d in enumerate(details):
            entry = {
                "sample_idx": i,
                "input_len": d.get("input_len"),
                "output_len": d.get("output_len"),
                "num_selected_kv": d.get("num_selected_kv"),
            }
            if d.get("input_len") and d.get("num_selected_kv"):
                entry["kv_select_ratio"] = round(d["num_selected_kv"] / d["input_len"], 4)
            if i < len(ds_scores_list):
                entry["score"] = round(ds_scores_list[i], 4)
            ds_sample_data.append(entry)
        with open(sample_file, "w") as f:
            json.dump(ds_sample_data, f, indent=2)

    # ===== Build lm_eval-style results table =====
    # Compute group scores
    group_scores = {}
    group_stderrs = {}
    for ds_name, score in scores.items():
        grp = _dataset2group.get(ds_name, "Other")
        if grp not in group_scores:
            group_scores[grp] = []
            group_stderrs[grp] = []
        group_scores[grp].append(score / 100.0)
        group_stderrs[grp].append(per_sample_scores.get(ds_name, {}).get("stderr", 0) / 100.0)
    for grp in group_scores:
        vals = group_scores[grp]
        group_scores[grp] = round(np.mean(vals), 4)
        group_stderrs[grp] = round(np.mean(group_stderrs[grp]), 4)

    n_samples = args.n if args.n > 0 else "all"

    # Build table lines
    task_lines = []
    group_lines = []
    # Sort groups
    group_order = ["Single-Document QA", "Multi-Document QA", "Summarization",
                   "Few-shot Learning", "Synthetic Tasks", "Code Completion"]
    for grp in group_order:
        if grp not in group_scores:
            continue
        task_lines.append({
            "task": f"- {grp}", "n_shot": "", "metric": "score",
            "value": group_scores[grp], "stderr": group_stderrs[grp], "is_group": True,
        })
        group_lines.append({
            "group": f"- {grp}", "value": group_scores[grp], "stderr": group_stderrs[grp],
        })
        # Add individual datasets under this group
        for ds_name in sorted(scores.keys()):
            if _dataset2group.get(ds_name) != grp:
                continue
            stderr_val = per_sample_scores.get(ds_name, {}).get("stderr", 0) / 100.0
            task_lines.append({
                "task": f" - longbench_{ds_name}", "n_shot": 0,
                "metric": _dataset2metric_name.get(ds_name, "score"),
                "value": scores[ds_name] / 100.0, "stderr": stderr_val, "is_group": False,
            })

    def _fmt_table(task_lines):
        lines = []
        hdr = f"|{'Tasks':^35}|{'n-shot':>6}|{'Metric':^20}|   |{'Value':>6} |   |{'Stderr':>6}|"
        sep = f"|{'-'*35}|{'-'*6}:|{'-'*20}|---|{'-'*6}:|---|{'-'*6}:|"
        lines.append(hdr)
        lines.append(sep)
        for t in task_lines:
            ns = str(t["n_shot"]) if t["n_shot"] != "" else ""
            lines.append(
                f"|{t['task']:<35}|{ns:>6}|{t['metric']:<20}|{'↑':^3}|{t['value']:>6.4f}|{'±':^3}|{t['stderr']:>6.4f}|"
            )
        return "\n".join(lines)

    def _fmt_group_table(group_lines):
        lines = []
        hdr = f"|{'Groups':^25}|{'Metric':^8}|   |{'Value':>6} |   |{'Stderr':>6}|"
        sep = f"|{'-'*25}|{'-'*8}|---|{'-'*6}:|---|{'-'*6}:|"
        lines.append(hdr)
        lines.append(sep)
        for g in group_lines:
            lines.append(
                f"|{g['group']:<25}|{'score':<8}|{'↑':^3}|{g['value']:>6.4f}|{'±':^3}|{g['stderr']:>6.4f}|"
            )
        return "\n".join(lines)

    # ===== Build summary text =====
    summary_parts = []
    summary_parts.append(f"{'='*70}")
    summary_parts.append(f"OmniKV LongBench Experiment Log")
    summary_parts.append(f"{'='*70}")
    summary_parts.append(f"")
    summary_parts.append(f"Timestamp:          {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_parts.append(f"Config:             {args.cfg}")
    summary_parts.append(f"Model:              {d_cfg.get('model_name', 'unknown')}")
    summary_parts.append(f"Model class:        {d_cfg.get('model_cls', 'unknown')}")
    summary_parts.append(f"Thinking mode:      {'ON' if enable_thinking else 'OFF'}")
    summary_parts.append(f"Max context len:    {d_cfg.get('max_context_len', -1)}")
    summary_parts.append(f"Samples/dataset:    {n_samples}")
    summary_parts.append(f"")
    summary_parts.append(f"--- Hardware ---")
    summary_parts.append(f"GPU:                {gpu_name}")
    summary_parts.append(f"Peak GPU memory:    {peak_gpu_mem:.2f} GB")
    summary_parts.append(f"Total elapsed time: {exp_total_time:.1f}s ({exp_total_time/60:.1f}min)")
    summary_parts.append(f"")
    summary_parts.append(f"--- KV Cache ---")
    if kv_cache_info:
        summary_parts.append(f"Per-layer ratio:    {kv_cache_info['per_layer_token_ratio']*100:.2f}%")
        summary_parts.append(f"Full cache layers:  {kv_cache_info['full_cache_layers']}/{kv_cache_info['total_layers']}")
        summary_parts.append(f"Sparse layers:      {kv_cache_info['sparse_cache_layers']}/{kv_cache_info['total_layers']}")
        summary_parts.append(f"Overall KV ratio:   {kv_cache_info['overall_kv_cache_ratio']*100:.2f}%")
        summary_parts.append(f"Select layers:      {d_cfg.get('do_select_layers', 'N/A')}")
        summary_parts.append(f"Selector:           {d_cfg.get('selector_cls', 'N/A')}")
    else:
        summary_parts.append(f"(not available)")
    summary_parts.append(f"")
    summary_parts.append(f"--- Per-Dataset Timing ---")
    for ds_name, stat in per_dataset_stats.items():
        summary_parts.append(f"  {ds_name:30s}  {stat['num_samples']:>4} samples  {stat['elapsed_time_s']:>8.1f}s  peak {stat['peak_gpu_memory_gb']:.2f}GB")
    summary_parts.append(f"")
    summary_parts.append(f"--- Results ---")
    summary_parts.append(f"")
    if task_lines:
        summary_parts.append(_fmt_table(task_lines))
        summary_parts.append(f"")
        summary_parts.append(_fmt_group_table(group_lines))
    else:
        summary_parts.append(f"  (no scores available)")
    summary_parts.append(f"")
    avg_score = round(np.mean(list(scores.values())), 2) if scores else 0
    summary_parts.append(f"Average score (all datasets): {avg_score}")
    summary_parts.append(f"")

    # Inference stats summary
    summary_parts.append(f"--- Inference Stats ---")
    for ds_name, details in sample_details.items():
        input_lens = [d["input_len"] for d in details if d.get("input_len") is not None]
        output_lens = [d["output_len"] for d in details if d.get("output_len") is not None]
        selected_kvs = [d["num_selected_kv"] for d in details if d.get("num_selected_kv") is not None]
        parts = [f"  {ds_name}:"]
        if input_lens:
            parts.append(f"input_len={np.mean(input_lens):.0f} [{min(input_lens)}-{max(input_lens)}]")
        if output_lens:
            parts.append(f"output_len={np.mean(output_lens):.0f} [{min(output_lens)}-{max(output_lens)}]")
        if selected_kvs:
            parts.append(f"selected_kv={np.mean(selected_kvs):.0f} [{min(selected_kvs)}-{max(selected_kvs)}]")
        if input_lens and selected_kvs:
            ratios = [s / i for s, i in zip(selected_kvs, input_lens) if i > 0]
            parts.append(f"ratio={np.mean(ratios):.4f}")
        summary_parts.append("  ".join(parts))
    summary_parts.append(f"")
    summary_parts.append(f"{'='*70}")

    summary_text = "\n".join(summary_parts)

    # Save summary.txt
    summary_txt_path = os.path.join(run_dir, "summary.txt")
    with open(summary_txt_path, "w") as f:
        f.write(summary_text)

    # Save summary.json
    exp_log = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config_path": args.cfg,
        "model_name": d_cfg.get("model_name", "unknown"),
        "model_cls": d_cfg.get("model_cls", "unknown"),
        "enable_thinking": enable_thinking,
        "num_samples_per_dataset": n_samples,
        "max_context_len": d_cfg.get("max_context_len", -1),
        "kv_cache": kv_cache_info,
        "scores": scores,
        "avg_score": avg_score,
        "per_sample_stderr": {k: v["stderr"] for k, v in per_sample_scores.items()},
        "group_scores": {k: round(v * 100, 2) for k, v in group_scores.items()},
        "per_dataset_stats": per_dataset_stats,
        "total_elapsed_time_s": round(exp_total_time, 2),
        "peak_gpu_memory_gb": round(peak_gpu_mem, 2),
        "gpu": gpu_name,
        "datasets_evaluated": list(scores.keys()),
    }
    summary_json_path = os.path.join(run_dir, "summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(exp_log, f, ensure_ascii=False, indent=4)

    # Print summary to stdout
    print(f"\n{summary_text}")
    print(f"\nLogs saved to: {run_dir}/")
    print(f"  summary.txt   - formatted results")
    print(f"  summary.json  - machine-readable results")
    print(f"  samples/      - per-sample details")
