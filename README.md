# Benchmarking Transformer Single Forward Pass on GPU

This repository provides a simplebenchmarking suite for measuring the latency of a single forward pass through a HuggingFace Transformer model on CUDA GPUs. It supports both the standard PyTorch implementation and an optional Liger kernel acceleration for causal‑language models.

## Key features

* Warmup iterations to eliminate cold‑start overhead
* Configurable batch sizes and sequence lengths
* Optional Liger kernel support for causal‑language models
* CSV output for downstream analysis and plotting

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/DandinPower/pytorch-cuda-benchmark.git
   cd pytorch-cuda-benchmark
   ```

2. **Create and activate a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

Use the `bench.py` script to run your benchmark. At minimum, you must specify:

* `--model`: a HuggingFace model name or local checkpoint path
* `--batch_sizes`: comma‑separated list of batch sizes (e.g. `1,4,8`)
* `--context_lengths`: comma‑separated list of sequence lengths (e.g. `128,512,1024`)
* `--warmup_iters`: number of warmup iterations before timing begins
* `--test_iters`: number of timed iterations to average over

Optional flags:

* `--output_csv`: output path for latency results (default: `latency_results.csv`)
* `--liger_kernel`: enable the Liger kernel variant for causal‑language models

### Example

```bash
bash example.sh
```

After completion, you will find a CSV file (`qwen_latency.csv` in the example) containing columns:

* `model_name`
* `batch_size`
* `context_length`
* `min_ms`
* `max_ms`
* `mean_ms`