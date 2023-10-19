---
layout: post
title:  "Scalable Language Model Inference on Multiple NVIDIA and AMD GPUs"
date:   2023-10-19 07:30:00 -0700
author:   MLC Community
notitle: true
---

* TOC
{:toc}

## TL;DR

Machine Learning Compilation ([MLC](https://github.com/mlc-ai/mlc-llm)) makes it possible to compile and deploy large-scale language models running on multi-GPU systems with support for NVIDIA and AMD GPUs with high performance. Specifically, we run 4-bit quantized Llama2-70B at 34.5 tok/sec on two NVIDIA RTX 4090 and 29.9 tok/sec on two AMD Radeon 7900XTX. The same solution also scales well beyond two GPUs. Our evaluation shows the performance consistently improves up to eight A10G GPUs when running Llama2-70B and CodeLlama 34B.

## Background

Demand for access to powerful GPUs exceeds supply, leading to their increasing cost and limited availability. A significant driver behind this demand is that the appetite for large-scale Large Language Models (LLMs) is strong within academia and industry. However, many of the largest models, such as Meta's Llama2-70B, face a bottleneck due to their size and cannot be accommodated on a single less-powerful GPU. To address this challenge and assist the GPU-poor during this scarcity era, an ideal LLM serving framework should be universal and capable of deploying on a broader spectrum of devices, highly performant, and able to scale up with the addition of more GPUs.

Machine learning compilation (MLC) techniques offer such a solution, which allow **universal deployment** on all viable GPUs. Backed by TVM Unity, the latest generation of Apache TVM, MLC LLM is capable of generating highly performant code across all GPU backends, such as CUDA, ROCm, Vulkan, OpenCL, Metal, and WebGPU, achieving **state-of-the-art performance** across NVIDIA, AMD, and Intel GPUs. Most importantly, unlike a traditional C++ compiler, it compiles for both single-node and **multi-GPU and distributed** use cases, as machine learning necessitates.

In this blog post, MLC LLM demonstrates multiple advantages in serving large-scale LLMs:

- Performance: It compiles and generates highly performant code for multi-GPU inference.
- Universal deployment: It enables universal deployment of LLMs onto GPUs across a variety of vendors and architectures.
- Scalability: It scales efficiently and provides more acceleration as the number of GPUs increases.

## MLC-Powered Multi-GPU Inference

This section demonstrates MLC LLM’s multi-GPU capability by answering the following questions via experiments:

- Performance: How fast is the solution? Is MLC LLM sufficiently performant to serve large-scale LLMs?
- Scalability: How well can we scale? Will the latency meaningfully improve with more GPUs added to the system?
- Universal deployment: Which GPU backends does MLC LLM support out of the box?

### Settings

We focus on auto-regressive decoding performance with batch one latency in this blog post because minimum latency is often an important consideration for application builders and a core aspect of good inference performance (prefill = 8, decoding = 256). Long-context prefilling and continuous batching performance are left for future blog posts.

**GPUs**. We have chosen GPUs from two representative groups: server- and consumer-class GPUs. In the server class, we use A100 (80GB) and A10G (24GB), and in consumer-class GPUs, we use NVIDIA’s RTX 4090 (24GB) and AMD Radeon 7900 XTX (24GB). All numbers are based on PCIe, not NVLink.

**Models**. We focus on sufficiently large models; in our case, CodeLlama-34B and Llama2-70B. The 4-bit quantized model could fit into 2 GPUs in all our experimental settings, while FP16 numbers are provided whenever VRAM is sufficient.

**Drivers and OS**. All the experiments are run on Ubuntu 22.04. For NVIDIA GPUs, we use CUDA 12.1, except for vLLM, which only supports CUDA 11.8 at the moment. For AMD GPUs, we use ROCm 5.7, the latest generation of ROCm.

### Performance

We first examine the single-batch decoding performance of the solution on two RTX 4090 GPUs. This configuration allows us to effectively work with Llama-70B using 4-bit setups. To provide a reference point, we also benchmarked other solutions in this setup. This post focuses on the optimal latency that a multi-GPU system could possibly achieve; the reference frameworks may not be optimized for a multi-GPU latency-focused scenario. Nevertheless, we include these reference results to help us gauge the performance of the multi-GPU solution.

<p align="center">
  <img src="/img/multi-gpu/figure-1.svg" width="50%">
  <figcaption>Figure 1. Single batch performance of 4-bit CodeLlama-34B and Llama2-70B on two NVIDIA RTX 4090.</figcaption>
</p>

On two RTX 4090 GPUs, we achieve 34 tok/sec on Llama2-70B and 64 tok/sec on CodeLlama-34B. It is remarkable that our solution enables significant performance using consumer GPUs, which is concretely accessible to a broader set of users who have limited access to high-end cloud GPUs.

### Scalability

The second question we ask is how the solution scales with the number of GPUs. The figure on the left below shows the performance of FP16 on A100-PCIe and A10G-PCIe, while the figure on the right shows the performance of 4-bit-quantized models on the same set of server-class GPUs. FP16 serving numbers are not provided for Exllama because it does not support FP16. Across all the experiments, we can observe that our solution continues to improve performance as we increase the number of GPUs.

<p align="center">
  <img src="/img/multi-gpu/figure-2.svg" width="100%">
  <figcaption>Figure 2. Scaling of fp16 and 4-bit CodeLlama-34 and Llama2-70B on A100-80G-PCIe and A10G-24G-PCIe, up to 8 GPUs, single batch.</figcaption>
</p>

It is worth mentioning that the scaling is not yet linear in this case. There are several factors that contribute to the result: First, the absence of NVLink. We noticed that PCIe and the host CPU may significantly contribute to the overhead and potential noise in our 8xA100 experiment. More specifically, LLM performance could degrade by up to 30% when the CPU is busy with tasks from other tenants, indicating that a much faster independent interconnect could potentially help with performance. Also, resource under-saturation. It is generally more challenging for auto-regressive decoding to saturate GPU resources. We noticed that the kernel speeds up less proportionally as workloads reduce on each GPU. For example, the multi-head attention kernel from cutlass operates at a similar latency with half of the attention heads in one of our experiments.

Nevertheless, having the ability to scale means we can achieve faster speeds or simply leverage multiple resource-constrained devices to serve even bigger models.

### Universal deployment: Support for Multi-AMD-GPU

There have been many LLM inference solutions since the bloom of open-source LLMs. Most of the performant inference solutions are based on CUDA and optimized for NVIDIA GPUs. Meanwhile, due to the high demand for compute availability and growing diversity of devices, it is useful to extend support to a broader class of hardware accelerators, with AMD being one potential candidate.

<p align="center">
  <img src="/img/multi-gpu/figure-3.svg" width="25%">
  <figcaption>Figure 3. Two-GPU single-batch inference: NVIDIA RTX 4090 vs AMD Radeon 7900 XTX on 4-bit Llama2-70B and CodeLlama-34B.</figcaption>
</p>

By adopting the universal deployment approach, MLC enables us to deploy on AMD GPUs through ROCm. We tested the same solution on two AMD 7900 XTX GPUs, and the results showed that these two AMD GPUs can achieve 30 tok/sec for Llama2-70B. This indicates that we can achieve approximately 85% of the results produced by two RTX 4090 GPUs. Considering that AMD GPUs cost $1000 per card, the setup with two AMD GPUs can be cost-effective for running Llama2-70B models. This result suggests that, empowered by MLC LLM, with the right price and availability, AMD GPUs could have viable performance/cost competitiveness.

Similar scalability is anticipated with more than two consumer GPUs, which we unfortunately do not have access to at the time of writing. In this case, the existing A10G experiment is a good proxy to scaling with NVIDIA and AMD consumer GPUs.

## Using MLC LLM

### Docker

The Dockerfile and corresponding instructions are provided in a dedicated GitHub [repo](https://github.com/mlc-ai/llm-perf-bench) to reproduce MLC LLM performance for both single-GPU and multi-GPU, CUDA and ROCm.

### Python API

The instructions below showcase how to use the multi-GPU feature in pure Python.

**Step 0. Install MLC LLM Python package**. Create an isolated conda virtual environment, and then install a nightly built of MLC LLM via pip following the [instructions](https://llm.mlc.ai/docs/install/mlc_llm.html):

```bash
VENV=mlc-llm-venv
conda create -n $VENV -c conda-forge python numpy pytorch-cpu scipy
conda activate $VENV
pip install .... # Instructions: https://llm.mlc.ai/docs/install/mlc_llm.html
```

**Step 1. Download pre-quantized weights**. The commands below download 4bit-quantized Llama2-70B from HuggingFace.

```bash
MODEL=Llama-2-70b-chat-hf

git lfs install
git clone https://huggingface.co/mlc-ai/mlc-chat-$MODEL-q4f16_1 ./dist/$MODEL-q4f16_1/params
```

**Step 2. Compile for multi-GPU inference**. Download MLC LLM’s Python-based compiler, and then compile the model with it. It may take a few seconds. Tweak "--num-shards" to set the number of GPUs to use:

```bash
git clone --recursive https://github.com/mlc-ai/mlc-llm
export PYTHONPATH=$(pwd)/mlc-llm/:$PYTHONPATH

curl https://raw.githubusercontent.com/mlc-ai/llm-perf-bench/main/model_configs/$MODEL.json \
     --create-dirs -o ./dist/models/$MODEL/config.json

python3 -m mlc_llm.build --build-model-only \
    --model ./dist/models/$MODEL/ \
    --quantization q4f16_1 \
    --max-seq-len 4096 \
    --num-shards 2 \ # e.g. 2, 4, 8
    --target cuda --use-cuda-graph
```

**Step 3. Run Llama2-70B using Python API**. Follow the snippet below to use the Python API for multi-GPU inference:

```python
from mlc_chat import ChatModule, ChatConfig
from mlc_chat.callback import StreamToStdout

cm = ChatModule(
  model="Llama-2-70b-chat-hf-q4f16_1",
  chat_config=ChatConfig(num_shards=2), # Number of GPUs
)
cm.generate(
    prompt="What is the meaning of life?",
    progress_callback=StreamToStdout(callback_interval=2),
)
```

## Discussion and Future works
**Machine Learning Compilation**. We leverage Apache TVM Unity, the latest machine learning compilation techniques that allow the representation and optimization of machine learning programs. MLC utilizes cross-layer representation and optimization for an end-to-end machine learning stack and makes use of lower-level compilers (LLVM/MLIR) and libraries to generate binary code. Specifically, we model multi-GPU inference using TVM Unity’s Single-Program-Multiple-Data (SPMD) representation with tensor parallelism. It further reduces collective library calls to NCCL/RCCL, highly optimized by NVIDIA and AMD. With MLC, we can conveniently represent pipelines, tensor parallelism, their lowering, and potential cross-layer optimization opportunities.

This post is part of the ongoing effort to bring high-performance universal deployment via MLC. We are also actively working on several areas to generalize our study:

- Enabling batching and speculation support;
- Integrating with the PyTorch ecosystem;
- Empowering more quantization and model architectures;
- Optimizations for long-context.

Our final takeaway is that machine learning system engineering is a continuous challenge because model progress is moving very fast and the others of the stack (HW and system software) also evolve. Therefore, effectiveness is not just about building “currently-working” solutions, but also about continuously bringing in the latest ML research and engineering techniques and adapting to new platforms. Productivity in machine learning engineering is crucial. Thanks to the Python-first ML compilation development flow in TVM Unity, we have built a universal multi-GPU solution seamlessly integrated with existing multi-backend code generation. We anticipate that related approaches will become even more valuable as we explore more ideas to achieve universal deployments and address the hardware availability problem.
