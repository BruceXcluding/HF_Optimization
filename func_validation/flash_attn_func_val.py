import math
import time
from einops import rearrange
import torch
import torch.nn.function as F
from flash_attn_interface import flash_attn_unpadded_func
from flash_attn_triton import flash_attn_func as attention
from xformers.ops import fmha, LowerTriangularMask
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--fnc", default="pt_native", required=True, type=str, choices=["pt_native", "fav1","fav2","xformers"], help="QKV function")
args = parser.parse_args()
fnc = args.fnc

def custom_attention(q,k,v,causal=False):
    score = torch.matmul(q,k.transpose(-2,-1))/math.sqrt(q.size(-1))
    if causal:
        mask = torch.triu(torch.ones(score.shape[-2],score.shape[-1]),diagonal=1)
        mask = mask.masked_fill(mask==1,torch.finfo(q.dtype).min)
        mask = mask.to(q.device,q.dtype)
        score = score + mask
    attn = F.softmax(score,dim=-1)
    o = torch.matmul(attn,v)
    return o

def pytorch_func(q,k,v,causal=False):
    o = F.scaled_dot_product_attention(q,k,v,is_causal=causal)
    return o

def flash_attentionv1(q,k,v,causal=False):
    bsz = q.shape[0]
    qv_seq_len = q.shape[1]
    kv_seq_len = k.shape[1]
    q_list,k_list = [0], [0]
    length_q, length_k = 0,0
    for i in range(bsz):
        length_q = length_q + qv_seq_len
        length_k = length_k + kv_seq_len
        q_list.append(length_q)
        k_list.append(length_k)

    q_cu_seq_lens = torch.as_tensor(q_list,device="cuda").int()
    k_cu_seq_lens = torch.as_tensor(k_list,device="cuda").int()
    q = q.flatten(0,1)
    k = k.flatten(0,1)
    v = v.flatten(0,1)

    o = flash_attn_unpadded_func(q,k,v,cu_seqlens_q=q_cu_seq_lens,cu_seqlens_k=k_cu_seq_lens,max_seqlen_q=qv_seq_len,max_seqlen_k=kv_seq_len,dropout_p=0,softmax_scale=None,causal=causal,return_attn_probs=False)
    return o

def flash_attentionv2(q,k,v,hd,causal=False):
    sm_scale = 1 / math.sqrt(hd)
    o = attention(q,k,v,causal,sm_scale)
    return o

def xformers_attention(q,k,v,hd,causal=False):
    xformers_attn_bias = LowerTriangularMask()
    scale = 1/math.sqrt(hd)
    attn_bias = xformers_attn_bias if causal else None
    o = fmha.memory_efficient_attention_forward(q,k,v,scale=scale,attn_bias=attn_bias,op=fmha.ck.FwOp)
    return o


def test(func_name,q,k,v,*args,**kwargs):
    if func_name in ["custom_attention","pytorch_func"]:
        q = rearrange(q,"a b c d -> a c b d")
        k = rearrange(k,"a b c d -> a c b d")
        v = rearrange(v,"a b c d -> a c b d")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    for _ in range(6):
        o = globals()[func_name](q,k,v,*args,**kwargs)
    torch.cuda.synchronize()
    st = time.time()
    o = globals()[func_name](q,k,v,*args,**kwargs)
    torch.cuda.synchronize()
    dt = time.time()-st
    max_memory = torch.cuda.max_memory_allocated()
    torch.cuda.empty_cache()

    if func_name in ["custom_attention","pytorch_func"]:
        o = rearrange(o,"a b c d -> a c b d") 

    return o, dt, max_memory

def run_test(fnc):
    bsz = [8,16]
    sql = [128,1024,2048,4096]
    nh = 32
    hd = 128
    d_type = torch.float16
    causal = [False,True]
    
    for c in causal:
        for b in bsz:
            for sl in sql:
                print(f"shape:({bsz},{sql},{nh},{hd},dtype:{dtype},causal:{causal})")
                q = torch.randn((bsz,sql,nh,hd)).to("cuda:0",dtype)
                k = torch.rand_like(q)
                v = torch.rand_like(q)

                o,t,m = test("custom_attention",q,k,v,causal=False)
                print(f"custom pytorch time:{t:.6f},peak memory: {m} MB")

                if fnc == "pt_native":
                    pf_o,pf_t,pf_m = test("pytorch_func",q,k,v,causal=False)
                    print(f"custom pytorch time:{pf_t:.6f},speedup:{t/pf_t:.2f};peak memory: {pf_m} MB, save: {int((m-pf_m)/m*100)}%")
                    assert torch.allclose(o,pf_o,rtol=1e-2,atol=1e-2)
                elif fnc == "fav1":
                    fav1_o,fav1_t,fav1_m = test("flash_attentionv1",q,k,v,causal=False)
                    print(f"custom pytorch time:{fav1_t:.6f},speedup:{t/fav1_t:.2f};peak memory: {fav1_m} MB,save: {int((m-fav1_m)/m*100)}%")
                    assert torch.allclose(o,fav1_o,rtol=1e-2,atol=1e-2)
                elif fnc == "fav2":
                    fav2_o,fav2_t,fav2_m = test("flash_attentionv2",q,k,v,hd,causal=False)
                    print(f"custom pytorch time:{fav2_t:.6f},speedup:{t/fav2_t:.2f};peak memory: {fav2_m} MB,save: {int((m-fav2_m)/m*100)}%")
                    assert torch.allclose(o,fav2_o,rtol=1e-2,atol=1e-2)
                elif fnc == "xformers":
                    xf_o,xf_t,xf_m = test("xformers_attention",q,k,v,hd,causal=False)
                    print(f"custom pytorch time:{xf_t:.6f},speedup:{t/xf_t:.2f};peak memory: {xf_m} MB,save: {int((m-xf_m)/m*100)}%")
                    assert torch.allclose(o,xf_o,rtol=1e-2,atol=1e-2)
                else:
                    raise Exception("Choose the function")
                 


if __name__ == "__main__":
    run_test(args.fnc)
