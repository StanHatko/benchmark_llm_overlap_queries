# Investigate Overlapping Text Effect on vLLM Performance

If multiple concurrent queries have large overlapping sections vs. not having such sections,
does this effect vLLM performance? Check this with some experiments.

First test is to detect random number in list, both with same list of numbers for each query vs.
different list for different queries. Can be generated automatically and easy to test.


## Test on Lambda Labs GPU Instance

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

Check time per run with 100 runs, with 1000 entries generated per run,
each list being different, 50 threads:

```bash
for i in $(seq 1 100);
do
    i=$(printf "%03d" $i)
    echo $i
    ./generate_detect_num_list.py /tmp/llm_test_diff_$i 1000 0
    time ( ls -1 /tmp/llm_test_diff_$i*.json | parallel -j 50 ./send_local_llm_query.py ) >>~/time_results_diff.txt
done
```

Time taken (in seconds) in 10 runs:

* 5.806

### Performance Test with Same

Test with 1000 entries generated, all same, 50 threas:

```bash
./generate_detect_num_list.py /tmp/llm_test_same_01 1000 1
time ( ls -1 /tmp/llm_test_basic_*.json | parallel -j 50 ./send_local_llm_query.py )
```

Time taken (in seconds) in 10 runs: 

* 5.559s
