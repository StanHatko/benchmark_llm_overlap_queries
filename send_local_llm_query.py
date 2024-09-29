#!/usr/bin/env python
"""
Send query to localhost VLLM server, for test purposes.
"""


import json

import fire
from openai import OpenAI


def send_query(prompt_file: str, output_file: str):
    """
    Send query to localhost LLM server and get response.
    """

    print("Load input file:", prompt_file)
    with open(prompt_file, "r", encoding="UTF8") as f:
        messages = json.load(f)

    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
    )
    completion = client.chat.completions.create(
        messages,
        model="hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4",
    )
    print("Completion result:", completion)
    breakpoint()


if __name__ == "__main__":
    fire.Fire(send_query)
