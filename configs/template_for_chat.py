from tiny_tools.log import logger, warning_once

prompt_template = {
#     "llama_cot": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
#
# {system_prompt}<|eot_id|>
# <|start_header_id|>user<|end_header_id|>
#
# {user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
#
# I will answer this question step by step.
# First, I have broken down this question into several sub-questions:
# 1.""",

    "llama_cot": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant. <|eot_id|>
<|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Evidence (step by step) then answer: 1.""",

    "llama_cot_2stage": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant. <|eot_id|>
<|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

First, for a precise answer, we can break down this multi-hop question into two to four sub-questions, as follows:""",

    "yi_cot": """You are helpful assistant who thinks step by step.
You will give your final answer in less than 80 words.
{user_message}
Evidence (less than 5 steps) then final answer: Step 1:""",

    'yi_cot_2stage': """You are helpful assistant who thinks step by step.
{user_message}

First, addition result is""",
    
    'yi': 'You are helpful assistant.\n{user_message}',

    'llama': """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""",

    'qwen3': """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
""",

    'qwen3_thinking': """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
<think>
""",

    'qwen3_no_thinking': """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
/no_think
{user_message}<|im_end|>
<|im_start|>assistant
""",
}


def get_chat_template(model_name, use_cot=False, enable_thinking=None):
    model_name = model_name.lower()
    if use_cot:
        warning_once(logger, "!!!!!!正在使用COT")

    if 'qwen3' in model_name or 'qwen-3' in model_name:
        if enable_thinking is True:
            warning_once(logger, "Qwen3: thinking mode ENABLED")
            return prompt_template['qwen3_thinking']
        elif enable_thinking is False:
            warning_once(logger, "Qwen3: thinking mode DISABLED")
            return prompt_template['qwen3_no_thinking']
        else:
            return prompt_template['qwen3']
    elif 'llama' in model_name:
        if use_cot:
            return prompt_template['llama_cot']
        else:
            return prompt_template['llama']
    elif 'yi' in model_name:
        if use_cot:
            return prompt_template['yi_cot']
        else:
            return prompt_template['yi']
    else:
        raise ValueError('Not Support')
