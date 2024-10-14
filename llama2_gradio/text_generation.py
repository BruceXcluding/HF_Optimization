import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from argparse import ArgumentParser
import os
import gc
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import gradio as gr

parser = ArgumentParser()
parser.add_argument("--mode", default="pt_native", required=True, type=str, choices=["pt_native", "fav2"], help="QKV mode")
parser.add_argument("--batch", default=8, type=int, choices=["1", "8"], help="")
parser.add_argument('--load_in_8bit',action='store_true',help='Use 8 bit quantified model')

args = parser.parse_args()
cache = True
saved_model = False
d_type = torch.float16
b = args.batch
qkv_m = args.mode
device_map = {"": 0}
max_memory = f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB'
load_in_8bit = args.load_in_8bit

model_id = "TheBloke/Llama-2-7B-fp16"

if torch.cuda.is_available():
    device = torch.device(0)
else:
    device = torch.device('cpu')

def llm_gen_tokens(model,slider,input_ids,tokenizer):
    return model.generate(
        **input_ids,
        do_sample=False,
        max_new_tokens=slider,
        use_cache=cache,
        pad_token_id=tokenizer.eos_token_id
    )


model = None
def llm_gen_setup():
    global model, tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=d_type, ignore_mismatched_sizes = True, qkv_mode=qkv_m,load_in_8bit=load_in_8bit,device_map=device_map,max_memory=max_memory)
    tokenizer = AutoTokenizer.from_pretrained(model_list, use_fast=False)

    model.eval()
        
def predict(input_sentences,slider):
    if len(input_sentences) >= 4096:
        print("OOM when seq_len >= 4k ")
    else:
        if b > len(input_sentences):
                input_sentences *= math.ceil(b / len(input_sentences))
        inputs = input_sentences[:b]
        input_ids = tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=False) 

         for t in input_ids:
            if torch.is_tensor(input_ids[t]):
                input_ids[t] = input_ids[t].to(device)
            
         gen_tokens = model.generate(**input_ids,do_sample=False,max_new_tokens=slider,use_cache=cache,pad_token_id=tokenizer.eos_token_id)
         outputs = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
    return outputs[0]

llm_gen_setup()

demo = gr.Interface(fn=predict,inputs=[gr.Textbox(label="Prompt"),gr.Slider(label="Max new tokens", value=32, maximum=1024,minimum=1)],outputs=[gr.Textbox(label="Generation")])

gr.close_all()
demo.queue().launch(share=True)
