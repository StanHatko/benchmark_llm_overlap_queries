#!/usr/bin/env python
"""
Generate LLM queries to see if number is in list.
Purpose is to benchmark if overlapping strings improve
speed, for instance by better using KV cache in VLLM.
"""

import json

import fire
import numpy as np


def generate_detect_num_list(
    out_file: str,
    num_gen: int,
    make_same: int,
):
    """
    Main function that runs the program.
    Due to how fire package works, make make_same int instead of bool.
    """

    queries = []
    check_list = None
    rgen = np.random.default_rng()

    for i in range(num_gen):
        if (make_same == 0) or (i == 0):
            check_list = rgen.integers(0, 1000, size=100)
            check_list_str = ", ".join([str(x) for x in check_list])

        rnum = int(rgen.integers(0, 1000))
        queries.append(
            [
                {
                    "role": "system",
                    "content": "Determine if the specified number is in the list.",
                },
                {
                    "role": "User",
                    "content": f"The list is: {check_list_str}.",
                },
                {
                    "role": "User",
                    "content": f"The number is: {rnum}.",
                },
            ],
        )

    with open(out_file, "w", encoding="UTF8") as f:
        json.dump(queries, f)


if __name__ == "__main__":
    fire.Fire(generate_detect_num_list)
