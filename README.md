# Investigate Overlapping Text Effect on vLLM Performance

If multiple concurrent queries have large overlapping sections vs. not having such sections,
does this affect vLLM performance? Check this with some experiments.

First test is to detect random number in list, both with same list of numbers for each query vs.
different list for different queries. Can be generated automatically and easy to test.


## Test on RunPod GPU Instance with 4-Bit Llama 3.3 70B

This test was done on a RunPod server with Nvidia A100 80 GB VRAM GPU.

The 4-bit AWQ quantized Llama 3.3 70B LLM was used for this test.

### Environment Setup

Setup the environment:

```bash
apt update
apt install parallel

python -m venv vllm-env
source vllm-env/bin/activate
python -m pip install -U pip setuptools wheel

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install vllm openai fire
```

Start vLLM server:

```bash
vllm serve lambdalabs/Llama-3.3-70B-Instruct-AWQ-4bit \
  --host 127.0.0.1 \
  --port 8000 \
  --max_model_len 16000 \
  --enable-prefix-caching \
  --enable-chunked-prefill
```

Initial test of LLM:

```bash
./generate_detect_num_list.py /tmp/llm_test_basic 10 0
./send_local_llm_query.py /tmp/llm_test_basic_000.json

ls -1 /tmp/llm_test_basic_*.json
ls -1 /tmp/llm_test_basic_*.json | parallel -j 10 ./send_local_llm_query.py
time ( ls -1 /tmp/llm_test_basic_*.json | parallel -j 10 ./send_local_llm_query.py )
```

### Performance Test with Different

Check time per run with 100 runs, with 100 entries generated per run,
each list being different, 50 threads:

```bash
./test_llm_detect_num_list_diff.sh >~/test_diff_log.txt 2>~/test_diff_time.txt
cat ~/test_diff_time.txt | grep real | perl -pe 's/.*0m//' | perl -pe 's/s$//'
```

The time results are in the file `time-taken-test2-diff.txt`.
The time in seconds has mean $TODO$ and standard deviation $TODO$.

### Performance Test with Same

Check time per run with 100 runs, with 100 entries generated per run,
within each run the list is same, 50 threads:

```bash
./test_llm_detect_num_list_same.sh >~/test_same_log.txt 2>~/test_same_time.txt
cat ~/test_same_time.txt | grep real | perl -pe 's/.*0m//' | perl -pe 's/s$//'
```

The time results are in the file `time-taken-test2-same.txt`.
The time in seconds has mean $TODO$ and standard deviation $TODO$.


## Test on Lambda Labs GPU Instance with 4-Bit Llama 3.1 8B (Old Test)

This test was done on a Lambda Labs gpu_1x_a100_sxm4 server in the Virginia, USA region.

The 4-bit GPTQ quantized Llama 3.1 8B LLM was used for this test.

### Environment Setup

Following steps done in conda environment with Python 3.12.
Without conda environment the vLLM server didn't work properly (errors with undefined symbols occurred).

Setup the environment:

```bash
sudo apt install parallel
pip install vllm openai fire
pip install --upgrade jinja2

git clone https://github.com/StanHatko/benchmark_llm_overlap_queries
cd benchmark_llm_overlap_queries
```

Start vLLM server:

```bash
vllm serve hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4 --host 127.0.0.1 --port 8000 --enable-prefix-caching
```

Initial test of LLM:

```bash
./generate_detect_num_list.py /tmp/llm_test_basic 10 0
./send_local_llm_query.py /tmp/llm_test_basic_000.json

ls -1 /tmp/llm_test_basic_*.json
ls -1 /tmp/llm_test_basic_*.json | parallel -j 10 ./send_local_llm_query.py
time ( ls -1 /tmp/llm_test_basic_*.json | parallel -j 10 ./send_local_llm_query.py )
```

### Performance Test with Different

Check time per run with 100 runs, with 100 entries generated per run,
each list being different, 50 threads:

```bash
./test_llm_detect_num_list_diff.sh >~/test_diff_log.txt 2>~/test_diff_time.txt
cat ~/test_diff_time.txt | grep real | perl -pe 's/.*0m//' | perl -pe 's/s$//'
```

The time results are in the file `time-taken-test1-diff.txt`.
The time in seconds has mean $6.45708$ and standard deviation $0.5997773$.

### Performance Test with Same

Check time per run with 100 runs, with 100 entries generated per run,
within each run the list is same, 50 threads:

```bash
./test_llm_detect_num_list_same.sh >~/test_same_log.txt 2>~/test_same_time.txt
cat ~/test_same_time.txt | grep real | perl -pe 's/.*0m//' | perl -pe 's/s$//'
```

The time results are in the file `time-taken-test1-same.txt`.
The time in seconds has mean $4.58922$ and standard deviation $0.591287$.
