# MMLU tester, with batch mode

This test RWKV models against MMLU, in batch mode.
With blank right padding, to help test the kernel accurately, internally.

This is not meant to be a replacement for the lm-eval-harness, and its more of an internal test tool.
This is not meant to be an efficient way to test MMLU. (it uses only a single GPU)

This was meant to just quickly test and debug the RWKV kernel.