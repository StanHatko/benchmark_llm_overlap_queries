# Investigate Overlapping Text Effect on vLLM Performance

If multiple concurrent queries have large overlapping sections vs. not having such sections,
does this effect vLLM performance? Check this with some experiments.

First test is to detect random number in list, both with same list of numbers for each query vs.
different list for different queries. Can be generated automatically and easy to test.


## Test on Lambda Labs GPU Instance

Setup the environment:

```bash
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
./send_local_llm_query.py /tmp/llm_test_basic_000.json /tmp/llm_test_basic_out_000.json
```
