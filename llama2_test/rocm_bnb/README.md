### ROCM GEMM TUNER

Quick Start

1. create a TGI docker image & launch a docker instance 

```bash
git clone --single-branch https://github.com/huggingface/text-generation-inference.git
cp Dockerfile_amd_bnb  text-generation-inference/.
cd text-generation-inference
docker build -f Dockerfile_amd_bnb -t tgi_bnb .

docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --name rocm5.7_bnb tgi_bnb:latest
```

2. install hipblaslt & bitsandbytes

```bash
pip install ninja joblib
git clone https://github.com/ROCmSoftwarePlatform/hipBLASLt.git
export PYTORCH_ROCM_ARCH=gfx90a
cd hipBLASLt
./install.sh -id --architecture gfx90a 
cd ..

conda install -c conda-forge gcc=12.1.0

git clone --single-branch --branch rocm_enabled https://github.com/ROCmSoftwarePlatform/bitsandbytes.git
cd bitsandbytes
make hip
python setup.py install

# To test if you have successfully installed 
python -m bitsandbytes

# To be benchmarks accuray benchmark from https://github.com/TimDettmers/bitsandbytes/issues/565
cd benchmarking/accuracy
python bnb_accuracy.py

#Accurate results should looks like
#tensor(526.7872, device='cuda:0')
#tensor(551.2297, device='cuda:0')
#tensor(574.9075, device='cuda:0')
#tensor(3435.1819, device='cuda:0')
#tensor(3480.1541, device='cuda:0') 
```

3. laucn, benchmark TGI

without gtuner
```bash
text-generation-launcher --model-id TheBloke/Llama-2-13B-Chat-fp16 &
text-generation-benchmark --tokenizer-name TheBloke/Llama-2-13B-Chat-fp16 --sequence-length 50 --decode-length 50 --runs 5
```

with bnb
```bash
text-generation-launcher --model-id TheBloke/Llama-2-13B-Chat-fp16 --quantize [bitsandbytes,bitsandbytes-nf4,bitsandbytes-fp4] &
text-generation-benchmark --tokenizer-name TheBloke/Llama-2-13B-Chat-fp16 --sequence-length 50 --decode-length 50 --runs 5
```

#### Without Bitsandbytes Results
```bash
| Step           | Batch Size | Average    | Lowest     | Highest    | p50        | p90        | p99        |
|----------------|------------|------------|------------|------------|------------|------------|------------|
| Prefill        | 1          | 69.99 ms   | 69.69 ms   | 70.41 ms   | 69.78 ms   | 70.41 ms   | 70.41 ms   |
|                | 2          | 77.72 ms   | 77.50 ms   | 77.83 ms   | 77.78 ms   | 77.83 ms   | 77.83 ms   |
|                | 4          | 99.46 ms   | 98.50 ms   | 101.25 ms  | 98.76 ms   | 101.25 ms  | 101.25 ms  |
|                | 8          | 135.25 ms  | 134.92 ms  | 135.75 ms  | 135.25 ms  | 135.75 ms  | 135.75 ms  |
|                | 16         | 232.22 ms  | 231.94 ms  | 232.56 ms  | 232.17 ms  | 232.56 ms  | 232.56 ms  |
|                | 32         | 410.07 ms  | 409.56 ms  | 410.38 ms  | 410.08 ms  | 410.38 ms  | 410.38 ms  |
| Decode (token) | 1          | 57.66 ms   | 57.65 ms   | 57.68 ms   | 57.66 ms   | 57.66 ms   | 57.66 ms   |
|                | 2          | 58.28 ms   | 58.27 ms   | 58.29 ms   | 58.29 ms   | 58.27 ms   | 58.27 ms   |
|                | 4          | 58.60 ms   | 58.59 ms   | 58.61 ms   | 58.60 ms   | 58.61 ms   | 58.61 ms   |
|                | 8          | 59.35 ms   | 59.34 ms   | 59.36 ms   | 59.35 ms   | 59.36 ms   | 59.36 ms   |
|                | 16         | 60.63 ms   | 60.61 ms   | 60.65 ms   | 60.63 ms   | 60.62 ms   | 60.62 ms   |
|                | 32         | 64.65 ms   | 64.64 ms   | 64.66 ms   | 64.66 ms   | 64.66 ms   | 64.66 ms   |
| Decode (total) | 1          | 2825.21 ms | 2824.62 ms | 2826.24 ms | 2825.19 ms | 2825.24 ms | 2825.24 ms |
|                | 2          | 2855.88 ms | 2855.48 ms | 2856.38 ms | 2856.17 ms | 2855.48 ms | 2855.48 ms |
|                | 4          | 2871.36 ms | 2870.76 ms | 2871.80 ms | 2871.37 ms | 2871.76 ms | 2871.76 ms |
|                | 8          | 2908.29 ms | 2907.88 ms | 2908.73 ms | 2908.28 ms | 2908.48 ms | 2908.48 ms |
|                | 16         | 2970.86 ms | 2970.13 ms | 2971.67 ms | 2971.14 ms | 2970.37 ms | 2970.37 ms |
|                | 32         | 3168.10 ms | 3167.46 ms | 3168.45 ms | 3168.11 ms | 3168.45 ms | 3168.45 ms |


| Step    | Batch Size | Average            | Lowest             | Highest            |
|---------|------------|--------------------|--------------------|--------------------|
| Prefill | 1          | 14.29 tokens/secs  | 14.20 tokens/secs  | 14.35 tokens/secs  |
|         | 2          | 25.73 tokens/secs  | 25.70 tokens/secs  | 25.81 tokens/secs  |
|         | 4          | 40.22 tokens/secs  | 39.50 tokens/secs  | 40.61 tokens/secs  |
|         | 8          | 59.15 tokens/secs  | 58.93 tokens/secs  | 59.29 tokens/secs  |
|         | 16         | 68.90 tokens/secs  | 68.80 tokens/secs  | 68.98 tokens/secs  |
|         | 32         | 78.04 tokens/secs  | 77.98 tokens/secs  | 78.13 tokens/secs  |
| Decode  | 1          | 17.34 tokens/secs  | 17.34 tokens/secs  | 17.35 tokens/secs  |
|         | 2          | 34.32 tokens/secs  | 34.31 tokens/secs  | 34.32 tokens/secs  |
|         | 4          | 68.26 tokens/secs  | 68.25 tokens/secs  | 68.27 tokens/secs  |
|         | 8          | 134.79 tokens/secs | 134.77 tokens/secs | 134.81 tokens/secs |
|         | 16         | 263.90 tokens/secs | 263.82 tokens/secs | 263.96 tokens/secs |
|         | 32         | 494.93 tokens/secs | 494.88 tokens/secs | 495.03 tokens/secs |
```


#### With Bitsandbytes Results

```bash
| Step           | Batch Size | Average    | Lowest     | Highest    | p50        | p90        | p99        |
|----------------|------------|------------|------------|------------|------------|------------|------------|
| Prefill        | 1          | 221.88 ms  | 221.68 ms  | 222.04 ms  | 221.88 ms  | 222.04 ms  | 222.04 ms  |
|                | 2          | 233.27 ms  | 233.10 ms  | 233.55 ms  | 233.21 ms  | 233.55 ms  | 233.55 ms  |
|                | 4          | 259.67 ms  | 259.42 ms  | 260.11 ms  | 259.61 ms  | 260.11 ms  | 260.11 ms  |
|                | 8          | 307.87 ms  | 307.60 ms  | 308.28 ms  | 307.78 ms  | 308.28 ms  | 308.28 ms  |
|                | 16         | 429.47 ms  | 429.08 ms  | 429.67 ms  | 429.60 ms  | 429.67 ms  | 429.67 ms  |
|                | 32         | 644.15 ms  | 643.67 ms  | 644.40 ms  | 644.25 ms  | 644.40 ms  | 644.40 ms  |
| Decode (token) | 1          | 174.18 ms  | 174.09 ms  | 174.24 ms  | 174.20 ms  | 174.18 ms  | 174.18 ms  |
|                | 2          | 177.07 ms  | 177.04 ms  | 177.13 ms  | 177.06 ms  | 177.04 ms  | 177.04 ms  |
|                | 4          | 177.69 ms  | 177.68 ms  | 177.75 ms  | 177.68 ms  | 177.68 ms  | 177.68 ms  |
|                | 8          | 178.78 ms  | 178.73 ms  | 178.85 ms  | 178.79 ms  | 178.75 ms  | 178.75 ms  |
|                | 16         | 181.13 ms  | 181.08 ms  | 181.20 ms  | 181.13 ms  | 181.14 ms  | 181.14 ms  |
|                | 32         | 185.58 ms  | 185.51 ms  | 185.62 ms  | 185.61 ms  | 185.62 ms  | 185.62 ms  |
| Decode (total) | 1          | 8534.74 ms | 8530.41 ms | 8537.62 ms | 8535.81 ms | 8534.91 ms | 8534.91 ms |
|                | 2          | 8676.34 ms | 8674.94 ms | 8679.19 ms | 8676.22 ms | 8674.94 ms | 8674.94 ms |
|                | 4          | 8706.93 ms | 8706.08 ms | 8709.69 ms | 8706.49 ms | 8706.17 ms | 8706.17 ms |
|                | 8          | 8760.28 ms | 8757.98 ms | 8763.65 ms | 8760.57 ms | 8758.82 ms | 8758.82 ms |
|                | 16         | 8875.47 ms | 8873.16 ms | 8878.72 ms | 8875.64 ms | 8875.71 ms | 8875.71 ms |
|                | 32         | 9093.49 ms | 9089.94 ms | 9095.28 ms | 9094.95 ms | 9095.28 ms | 9095.28 ms |


| Step    | Batch Size | Average            | Lowest             | Highest            |
|---------|------------|--------------------|--------------------|--------------------|
| Prefill | 1          | 4.51 tokens/secs   | 4.50 tokens/secs   | 4.51 tokens/secs   |
|         | 2          | 8.57 tokens/secs   | 8.56 tokens/secs   | 8.58 tokens/secs   |
|         | 4          | 15.40 tokens/secs  | 15.38 tokens/secs  | 15.42 tokens/secs  |
|         | 8          | 25.99 tokens/secs  | 25.95 tokens/secs  | 26.01 tokens/secs  |
|         | 16         | 37.26 tokens/secs  | 37.24 tokens/secs  | 37.29 tokens/secs  |
|         | 32         | 49.68 tokens/secs  | 49.66 tokens/secs  | 49.71 tokens/secs  |
| Decode  | 1          | 5.74 tokens/secs   | 5.74 tokens/secs   | 5.74 tokens/secs   |
|         | 2          | 11.30 tokens/secs  | 11.29 tokens/secs  | 11.30 tokens/secs  |
|         | 4          | 22.51 tokens/secs  | 22.50 tokens/secs  | 22.51 tokens/secs  |
|         | 8          | 44.75 tokens/secs  | 44.73 tokens/secs  | 44.76 tokens/secs  |
|         | 16         | 88.33 tokens/secs  | 88.30 tokens/secs  | 88.36 tokens/secs  |
|         | 32         | 172.43 tokens/secs | 172.40 tokens/secs | 172.50 tokens/secs |
```
#### With Bitsandbytes-nf4 Results
```bash
| Step           | Batch Size | Average    | Lowest     | Highest    | p50        | p90        | p99        |
|----------------|------------|------------|------------|------------|------------|------------|------------|
| Prefill        | 1          | 131.68 ms  | 131.44 ms  | 132.10 ms  | 131.63 ms  | 132.10 ms  | 132.10 ms  |
|                | 2          | 139.62 ms  | 138.94 ms  | 140.91 ms  | 139.33 ms  | 140.91 ms  | 140.91 ms  |
|                | 4          | 160.24 ms  | 159.95 ms  | 160.84 ms  | 160.16 ms  | 160.84 ms  | 160.84 ms  |
|                | 8          | 199.67 ms  | 198.79 ms  | 200.07 ms  | 199.80 ms  | 200.07 ms  | 200.07 ms  |
|                | 16         | 302.76 ms  | 302.39 ms  | 303.05 ms  | 302.88 ms  | 303.05 ms  | 303.05 ms  |
|                | 32         | 477.14 ms  | 476.69 ms  | 478.32 ms  | 476.86 ms  | 478.32 ms  | 478.32 ms  |
| Decode (token) | 1          | 117.37 ms  | 117.28 ms  | 117.42 ms  | 117.39 ms  | 117.39 ms  | 117.39 ms  |
|                | 2          | 117.98 ms  | 117.95 ms  | 118.00 ms  | 118.00 ms  | 117.95 ms  | 117.95 ms  |
|                | 4          | 118.46 ms  | 118.40 ms  | 118.49 ms  | 118.47 ms  | 118.48 ms  | 118.48 ms  |
|                | 8          | 119.01 ms  | 118.92 ms  | 119.19 ms  | 119.01 ms  | 118.98 ms  | 118.98 ms  |
|                | 16         | 120.13 ms  | 120.09 ms  | 120.18 ms  | 120.14 ms  | 120.14 ms  | 120.14 ms  |
|                | 32         | 124.33 ms  | 124.23 ms  | 124.39 ms  | 124.34 ms  | 124.39 ms  | 124.39 ms  |
| Decode (total) | 1          | 5751.28 ms | 5746.95 ms | 5753.53 ms | 5752.07 ms | 5751.99 ms | 5751.99 ms |
|                | 2          | 5780.94 ms | 5779.58 ms | 5782.23 ms | 5781.89 ms | 5779.58 ms | 5779.58 ms |
|                | 4          | 5804.51 ms | 5801.53 ms | 5806.00 ms | 5804.94 ms | 5805.39 ms | 5805.39 ms |
|                | 8          | 5831.64 ms | 5827.36 ms | 5840.33 ms | 5831.55 ms | 5830.09 ms | 5830.09 ms |
|                | 16         | 5886.56 ms | 5884.36 ms | 5888.81 ms | 5886.89 ms | 5886.63 ms | 5886.63 ms |
|                | 32         | 6092.34 ms | 6087.52 ms | 6095.04 ms | 6092.53 ms | 6095.04 ms | 6095.04 ms |


| Step    | Batch Size | Average            | Lowest             | Highest            |
|---------|------------|--------------------|--------------------|--------------------|
| Prefill | 1          | 7.59 tokens/secs   | 7.57 tokens/secs   | 7.61 tokens/secs   |
|         | 2          | 14.33 tokens/secs  | 14.19 tokens/secs  | 14.39 tokens/secs  |
|         | 4          | 24.96 tokens/secs  | 24.87 tokens/secs  | 25.01 tokens/secs  |
|         | 8          | 40.07 tokens/secs  | 39.99 tokens/secs  | 40.24 tokens/secs  |
|         | 16         | 52.85 tokens/secs  | 52.80 tokens/secs  | 52.91 tokens/secs  |
|         | 32         | 67.07 tokens/secs  | 66.90 tokens/secs  | 67.13 tokens/secs  |
| Decode  | 1          | 8.52 tokens/secs   | 8.52 tokens/secs   | 8.53 tokens/secs   |
|         | 2          | 16.95 tokens/secs  | 16.95 tokens/secs  | 16.96 tokens/secs  |
|         | 4          | 33.77 tokens/secs  | 33.76 tokens/secs  | 33.78 tokens/secs  |
|         | 8          | 67.22 tokens/secs  | 67.12 tokens/secs  | 67.27 tokens/secs  |
|         | 16         | 133.18 tokens/secs | 133.13 tokens/secs | 133.23 tokens/secs |
|         | 32         | 257.37 tokens/secs | 257.26 tokens/secs | 257.58 tokens/secs |

```

#### With Bitsandbytes-fp4 Results
```bash
| Step           | Batch Size | Average    | Lowest     | Highest    | p50        | p90        | p99        |
|----------------|------------|------------|------------|------------|------------|------------|------------|
| Prefill        | 1          | 120.42 ms  | 119.71 ms  | 122.25 ms  | 119.99 ms  | 122.25 ms  | 122.25 ms  |
|                | 2          | 127.53 ms  | 127.18 ms  | 127.79 ms  | 127.55 ms  | 127.79 ms  | 127.79 ms  |
|                | 4          | 148.20 ms  | 148.03 ms  | 148.34 ms  | 148.15 ms  | 148.34 ms  | 148.34 ms  |
|                | 8          | 187.68 ms  | 187.45 ms  | 187.95 ms  | 187.69 ms  | 187.95 ms  | 187.95 ms  |
|                | 16         | 287.36 ms  | 287.11 ms  | 287.80 ms  | 287.33 ms  | 287.80 ms  | 287.80 ms  |
|                | 32         | 470.53 ms  | 470.21 ms  | 471.17 ms  | 470.32 ms  | 471.17 ms  | 471.17 ms  |
| Decode (token) | 1          | 105.82 ms  | 105.75 ms  | 105.89 ms  | 105.84 ms  | 105.81 ms  | 105.81 ms  |
|                | 2          | 106.25 ms  | 106.20 ms  | 106.30 ms  | 106.27 ms  | 106.23 ms  | 106.23 ms  |
|                | 4          | 106.59 ms  | 106.53 ms  | 106.69 ms  | 106.58 ms  | 106.57 ms  | 106.57 ms  |
|                | 8          | 107.17 ms  | 107.11 ms  | 107.23 ms  | 107.15 ms  | 107.23 ms  | 107.23 ms  |
|                | 16         | 108.40 ms  | 108.38 ms  | 108.43 ms  | 108.39 ms  | 108.43 ms  | 108.43 ms  |
|                | 32         | 112.65 ms  | 112.61 ms  | 112.71 ms  | 112.64 ms  | 112.71 ms  | 112.71 ms  |
| Decode (total) | 1          | 5185.36 ms | 5181.71 ms | 5188.47 ms | 5186.20 ms | 5184.84 ms | 5184.84 ms |
|                | 2          | 5206.47 ms | 5203.81 ms | 5208.64 ms | 5207.47 ms | 5205.23 ms | 5205.23 ms |
|                | 4          | 5222.78 ms | 5219.79 ms | 5227.89 ms | 5222.52 ms | 5221.84 ms | 5221.84 ms |
|                | 8          | 5251.12 ms | 5248.36 ms | 5254.45 ms | 5250.18 ms | 5254.45 ms | 5254.45 ms |
|                | 16         | 5311.41 ms | 5310.76 ms | 5312.97 ms | 5311.11 ms | 5312.97 ms | 5312.97 ms |
|                | 32         | 5519.69 ms | 5517.73 ms | 5522.95 ms | 5519.20 ms | 5522.95 ms | 5522.95 ms |


| Step    | Batch Size | Average            | Lowest             | Highest            |
|---------|------------|--------------------|--------------------|--------------------|
| Prefill | 1          | 8.30 tokens/secs   | 8.18 tokens/secs   | 8.35 tokens/secs   |
|         | 2          | 15.68 tokens/secs  | 15.65 tokens/secs  | 15.73 tokens/secs  |
|         | 4          | 26.99 tokens/secs  | 26.96 tokens/secs  | 27.02 tokens/secs  |
|         | 8          | 42.63 tokens/secs  | 42.56 tokens/secs  | 42.68 tokens/secs  |
|         | 16         | 55.68 tokens/secs  | 55.59 tokens/secs  | 55.73 tokens/secs  |
|         | 32         | 68.01 tokens/secs  | 67.92 tokens/secs  | 68.05 tokens/secs  |
| Decode  | 1          | 9.45 tokens/secs   | 9.44 tokens/secs   | 9.46 tokens/secs   |
|         | 2          | 18.82 tokens/secs  | 18.81 tokens/secs  | 18.83 tokens/secs  |
|         | 4          | 37.53 tokens/secs  | 37.49 tokens/secs  | 37.55 tokens/secs  |
|         | 8          | 74.65 tokens/secs  | 74.60 tokens/secs  | 74.69 tokens/secs  |
|         | 16         | 147.61 tokens/secs | 147.56 tokens/secs | 147.62 tokens/secs |
|         | 32         | 284.07 tokens/secs | 283.91 tokens/secs | 284.17 tokens/secs |

```



