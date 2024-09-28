# Investigate Overlapping Text Effect on vLLM Performance

If multiple concurrent queries have large overlapping sections vs. not having such sections,
does this effect vLLM performance? Check this with some experiments.

First test is to detect random number in list, both with same list of numbers for each query vs.
different list for different queries. Can be generated automatically and easy to test.
