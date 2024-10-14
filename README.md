# HF_Optimization

This project is to demonstrate the complete test and application development process.

## Directory Structure

The following code lists the directory structure of HF_Optimization:

```
/func_validation/ : different flash attention implematation with ROCM optimized kernels or pytorch compared with customer's
/llama2_gradio/ : Llama model with gradio AI text generation UI applications
/llama2_test/
| | model/
| | | flash_attn/ : Llama model example with flash attention
| | | xformers/ : Llama model example with xformers
| | rocm_flash_attn/ : LLM model execution scripts with flash attention
| | rocm_xformers/ : LLM model execution scripts with xformers

```

## Environment Preparation

### Docker built up
```bash
alias drun='docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $HOME/ROCm:/ROCm'
drun --name hf_optimization rocm/pytorch:rocm5.7_ubuntu22.04_py3.10_pytorch_2.0.1
cd /ROCm
```
### Huggingface Transformers built up

#### Install HF transformers
```bash
pip install transformers==4.34.1
```
> **_Notes:_** **_Flash attention and xformers libraries are depended on different torch version. They need to be tested separately._**

#### Option 1. FlashAttention Installation

##### (1) reinstall the latest triton and flash-attention. 

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm5.7
pip uninstall pytorch-triton-rocm triton
git clone https://github.com/ROCmSoftwarePlatform/triton
cd triton/
git submodule update --init --recursive
cd python
python setup.py install

git clone https://github.com/ROCmSoftwarePlatform/flash-attention/ -b FA_triton_mosaic_kernels
cd flash-attention
git submodule update --init --recursive
```
##### (2) modify the flash attention code
```bash
cd {YOUR_ABS_PATH}/flash-attention/flash_attn
sed -i 's/bfloat16/float16/g' flash_attn_triton.py
```
##### (3) modify llama model files in HF transformers lib
refer to [**flash_attn**](./llama2_test/model/flash_attn), use the new configuration and modeling scripts
```bash
cp * {YOUR_ABS_PATH}/python3.10/site-packages/transformers/models/llama/.
```

#### Option 2. Xformers Installation

##### (1) reinstall the torch and xformers. 

```bash
pip uninstall torch torchvision torchaudio
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
git clone https://github.com/ROCmSoftwarePlatform/xformers
cd xformers
git submodule update --init --recursive
pip install -e ./
python -m xformers.info
```

##### (2) modify llama model files in HF transformers lib
refer to [**xformers**](./llama2_test/model/xformers), use the new configuration and modeling scripts
```bash
cp * {YOUR_ABS_PATH}/python3.10/site-packages/transformers/models/llama/.
```

#### Install bitsandbytes-rocm

##### compilation quickstart

```bash
git clone https://github.com/Lzy17/bitsandbytes-rocm
cd bitsandbytes-rocm

make hip
python setup.py install

#to test if you have successfully installed
python -m bitsandbytes

#To be benchmarks accuray benchmark from https://github.com/TimDettmers/bitsandbytes/issues/565
cd benchmarking/accuracy
python bnb_accuracy.py

#Accurate results should looks like
#tensor(526.7872, device='cuda:0')
#tensor(551.2297, device='cuda:0')
#tensor(574.9075, device='cuda:0')
#tensor(3435.1819, device='cuda:0')
#tensor(3480.1541, device='cuda:0')

#
```

## Module 1 - Functions Validation on INSTINCT/RADEON AI (GPU)

This part is to verify the functionality of the different flash attention implementation.

##### 1. PYTORCH FUNCTION (scaled_dot_product_attention)

```bash
cd ./func_validation
python flash_attn_func_val.py --fnc pt_native
```

##### 2. ROCM FLASH ATTENTION V1

> **_Notes:_** **_make sure switch to flash attention environment refer to previous part_**

```bash
cd ./func_validation
PYTHONPATH={YOUR_ABS_PATH}/flash-attention/flash_attn python flash_attn_func_val.py  --fnc fav1
```

##### 3. ROCM FLASH ATTENTION V2 
> **_Notes:_** **_make sure switch to flash attention environment refer to previous part_**

```bash
cd ./func_validation
PYTHONPATH={YOUR_ABS_PATH}/flash-attention/flash_attn python flash_attn_func_val.py  --fnc fav2
```

##### 4. ROCM Xformers Memory Efficiency 
> **_Notes:_** **_make sure switch to xformers environment refer to previous part_**

```bash
cd ./func_validation
python flash_attn_func_val.py --fnc xformers
```

## Module 2 - Llama2 Test on INSTINCT/RADEON AI (GPU)

Using the re-constructed llama2 model based on transformers library to do a text generation inference task.

### 1. ROCM FLASH ATTENTION V2

> **_Notes:_** **_make sure switch to flash attention environment refer to previous part_**

##### Run llama inference (--mode pt_native: torch native QKV, --mode fav2: flash attention v2)

```bash
cd ./llama2_test/rocm_flash_attn
PYTHONPATH={YOUR_ABS_PATH}/flash-attention/flash_attn python inference_llama.py  --mode pt_native
PYTHONPATH={YOUR_ABS_PATH}/flash-attention/flash_attn python inference_llama.py  --mode fav2
```

##### Run llama with bnb int8 quant inference (--mode pt_native: torch native QKV, --mode fav2: flash attention v2)

```bash
cd ./llama2_test/rocm_flash_attn
PYTHONPATH={YOUR_ABS_PATH}/flash-attention/flash_attn python inference_llama_quant.py  --mode pt_native
PYTHONPATH={YOUR_ABS_PATH}/flash-attention/flash_attn python inference_llama_quant.py  --mode fav2
```

### 2. ROCM Xformers Memory Efficiency

> **_Notes:_** **_make sure switch to xformers environment refer to previous part_**

##### Run llama inference (--mode pt_native: torch native QKV, --mode xformers: memory efficient attention)

```bash
cd ./llama2_test/rocm_xformers
python inference_llama.py  --mode pt_native
python inference_llama.py  --mode xformers
```
##### Run llama with bnb int8 quant inference (--mode pt_native: torch native QKV, --mode xformers: memory efficient attention)

```bash
cd ./llama2_test/rocm_xformers
python inference_llama_quant.py  --mode pt_native
python inference_llama_quant.py  --mode xformers
```

## Module 3 - Gradio Application on INSTINCT/RADEON AI (GPU)

Using the modified llama2 model based on transformers library and gradio library to build up an AI text generation UI application.

### 1. ROCM FLASH ATTENTION V2

> **_Notes:_** **_make sure switch to flash attention environment refer to previous part_**

##### Installation

```bash
pip install gradio accelerate sentencepiece
```

##### Run text generation app (--mode pt_native: torch native QKV, --mode fav2: flash attention v2)

```bash
cd ./llama2_gradio
PYTHONPATH={YOUR_ABS_PATH}/flash-attention/flash_attn python text_generation.py  --mode pt_native
PYTHONPATH={YOUR_ABS_PATH}/flash-attention/flash_attn python text_generation.py --mode fav2
```

##### Run chatbot app (--mode pt_native: torch native QKV, --mode fav2: flash attention v2)

```bash
cd ./llama2_gradio
PYTHONPATH={YOUR_ABS_PATH}/flash-attention/flash_attn python chatbot.py  --mode pt_native
PYTHONPATH={YOUR_ABS_PATH}/flash-attention/flash_attn python chatbot.py  --mode fav2
```




