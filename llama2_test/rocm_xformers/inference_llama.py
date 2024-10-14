import torch
import time
import math
import os
import gc
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.prompt_sample import input_select
from utils.arguments import parser
args = parser.parse_args()

# ======== Start: Parameters for LLM inf
d_type = torch.float16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # assume single-gpu
warm_up = True
cache = True
tc_enable = False
tc_mode = "default"
iters = 3
warmup = 2
seq_len = [1,128, 1024, 2048, 4096]
disp = False
saved_model = False
# ======== End: Parameters for LLM inf
str_space = "       "

model_lists = [
    "TheBloke/Llama-2-7B-fp16",
    ]

def print_perf_stats(latency_set, config, dtype, batch_size, warmup, mem_set):
    latency_set = list(latency_set)
    mem_set = list(mem_set)
    latency_set = latency_set[warmup:]
    mem_set = mem_set[warmup:]
    count = len(latency_set)

    if count > 0:
        latency_set.sort()
        latency_avg = sum(latency_set) / count
        mem_avg = sum(mem_set) / count

        num_layers = getattr(config, "num_layers", config.num_hidden_layers)
        num_parameters = num_layers * config.hidden_size * config.hidden_size * 12
        if dtype == torch.float16:
            num_bytes = 2
        if dtype == torch.float32:
            num_bytes = 4
        else:
            num_bytes = 1
        print(str_space + "Avg Latency/Tkn: {0:8.2f} ms".format(latency_avg * 1e3))
        print(str_space + "Avg BW:          {0:8.2f} GB/s".format(1/latency_avg * num_parameters * num_bytes / 1e9))
        print(str_space + "Avg flops:       {0:8.2f} GFlops/s".format(1/latency_avg * num_parameters * num_bytes * batch_size / 1e9))
        print(str_space + "Avg GPU mem:     {0:8.2f} GB".format(mem_avg))

def env_print(model, model_list):
    if tc_enable == True:
        print("[INFO] mode:    torch compilation with a mode: ", tc_mode)
    else:
        print("[INFO] mode:    torch eager")
    print("[INFO] torch:  ",torch.__version__)
    print("[INFO] rocm:   ", torch._C._cuda_getCompiledVersion())
    print("[INFO] device: ", torch.cuda.get_device_name(0))
    print("[INFO] model:  " + model_list, " -head_num: ", str(model.config.num_attention_heads), " -head_dim: ", str(int(model.config.hidden_size/model.config.num_attention_heads)), " -layers_num: " + str(model.config.num_hidden_layers))

def llm_gen_tokens(model, max_length, input_ids, tokenizer):
    return model.generate(
        **input_ids,
        do_sample=False,
        max_new_tokens=max_length,
        use_cache=cache,
        pad_token_id=tokenizer.eos_token_id
    )

model = None
def llm_gen_setup(qkv_m):
    for model_list in model_lists:
        model_dev = model_list.split("/")[0]
        model_name = model_list.split("/")[1]
        global model
        if model is not None:
            del model
        if model_dev == "TheBloke":
            # from github_repo
            if saved_model:
                model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=d_type, ignore_mismatched_sizes = True).cuda(device)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_list, torch_dtype=d_type, ignore_mismatched_sizes = True, qkv_mode = qkv_m).cuda(device)
            if tc_enable == True:
                for i in range(model.config.num_hidden_layers):
                    model.model.layers[i].self_attn = torch.compile(model.model.layers[i].self_attn, backend="inductor", mode= tc_mode)
            tokenizer = AutoTokenizer.from_pretrained(model_list, use_fast=False)
        else:
            raise Exception("check model name")

        model.eval()
        env_print(model, model_list)

        print("")
        print("[MODE] Prefill QKV mode: ", qkv_m)
        if model_name == "Llama-2-7B-fp16":
            batch_in_out = {1:32,8:32,16:32}
        else:
            raise Exception("check model name")

        assert warmup > 1, "warmup should be larger than 1"
        assert iters > warmup, "iters should be larger than warmup"
        model.config.qkv_mode = qkv_m
        model.config.qkv_mode = qkv_m

        for sl in seq_len:
            input_sentences = input_select(sl)
            for b, v in batch_in_out.items():
                gc.collect()
                if sl >= 4096 and b >= 16:
                    print("OOM when seq_len >= 4k and batch_size >= 16")
                else:
                    print("[INFO] resetting peak mem stats")
                    torch.cuda.reset_peak_memory_stats()
                    if b > len(input_sentences):
                        input_sentences *= math.ceil(b / len(input_sentences))
                    print(str_space + "batch_size: ", b, " seq_len: ", sl, " output_tokens: ", v)
                    inputs = input_sentences[:b]
                    input_ids = tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=False)
                    
                    for t in input_ids:
                        if torch.is_tensor(input_ids[t]):
                            input_ids[t] = input_ids[t].to(device)

                    latency_set = []
                    mem_set = []
                    for itr in range(iters):
                        start = time.perf_counter()
                        gen_tokens = llm_gen_tokens(model, v, input_ids, tokenizer)
                        tot_latency = time.perf_counter() - start
                        outputs = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
                        latency_set.append(tot_latency)
                        mem_set.append(torch.cuda.max_memory_allocated()/1e9)
                    print_perf_stats(map(lambda t: t / v, latency_set), model.config, d_type, b, warmup, mem_set)
                    if disp == True:
                        print("[INPUT]")
                        print(inputs)
                        print("[OUTPUT]")
                        print(outputs)
        print(" ")

if __name__ == "__main__":
    llm_gen_setup(args.mode)
