# Investigate Overlapping Text Effect on vLLM Performance

If multiple concurrent queries have large overlapping sections vs. not having such sections,
does this effect vLLM performance? Check this with some experiments.

First test is to detect random number in list, both with same list of numbers for each query vs.
different list for different queries. Can be generated automatically and easy to test.


## Test on Lambda Labs GPU Instance with 4-Bit Llama 3.1 8B

This test was done on a Lambda Labs gpu_1x_a100_sxm4 server in the Arizona, USA region.

The 4-bit GPTQ quantized Llama 3.1 8B LLM was used for this test.

### Environment Setup

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
vllm serve hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4 --host 127.0.0.1 --port 8000
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

The time results are in the file `time_taken_4bit_diff.txt`.
The time in seconds has mean 5.44607 and standard deviation 1.391389.

### Performance Test with Same

Check time per run with 100 runs, with 100 entries generated per run,
within each run the list is same, 50 threads:

```bash
./test_llm_detect_num_list_same.sh >~/test_same_log.txt 2>~/test_same_time.txt
cat ~/test_same_time.txt | grep real | perl -pe 's/.*0m//' | perl -pe 's/s$//'
```

The time results are in the file `time_taken_4bit_same.txt`.
The time in seconds has mean 5.38266 and standard deviation 1.253483.
