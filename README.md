# Introduction

This repo tries to benchmark the following implementations of
transducer loss in terms of speed and memory consumption:

- [k2][k2]
- [torchaudio][torchaudio]
- [optimized_transducer][optimized_transducer]

The benchmark results are saved in <https://huggingface.co/csukuangfj/transducer-loss-benchmarking>

# Environment setup

## Install torchaudio
Please refer to <https://github.com/pytorch/audio> to install torchaudio.
Note: It requires torchaudio >= 0.10.0

## Install k2

Please refer to <https://k2-fsa.github.io/k2/installation/index.html> to install k2.
Note: It requires at least k2 v1.13.

**Caution**: Please install a version that is compiled against the version of the PyTorch
you currently have locally.

## Install PyTorch Profiler TensorBoard Plugin


A simple way to install it is

```bash
pip install torch-tb-profiler
```

Please refer to <https://github.com/pytorch/kineto/tree/main/tb_plugin> for other alternatives.


# Steps to get the benchmark results


**TODOs**
- [ ] Add benchmark results for [warp-transducer][warp-transducer] and [warprnnt_numba][warprnnt_numba]
- [ ] ... ...



[k2]: http://github.com/k2-fsa/k2
[torchaudio]: https://github.com/pytorch/audio
[optimized_transducer]: https://github.com/csukuangfj/optimized_transducer
[warp-transducer]: https://github.com/HawkAaron/warp-transducer
[warprnnt_numba]: https://github.com/titu1994/warprnnt_numba
