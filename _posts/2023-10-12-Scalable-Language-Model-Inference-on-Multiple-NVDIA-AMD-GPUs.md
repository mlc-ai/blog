---
layout: post
title:  "Scalable Language Model Inference on Multiple NVIDIA and AMD GPUs"
date:   2023-10-12 09:30:00 -0400
author:   MLC Community
notitle: true
---

* TOC
{:toc}

## TL;DR

Machine Learning Compilation (MLC) makes it possible to compile and deploy large-scale language models on multiple NVIDIA and AMD GPUs with competitive performance. Specifically, we run 4-bit quantized Llama2-70B at 34.5 tok/sec on two NVIDIA RTX 4090 and 29.9 tok/sec on two AMD Radeon 7900XTX. We also observe that the solution consistently scales with more GPUs.

## Background

We are currently facing a global shortage of powerful GPUs, leading to their increasing cost and limited availability. Despite this, the demand for large-scale Large Language Models (LLMs) remains strong within academia and industry. However, many of the largest models, such as Meta's Llama2-70B, face a bottleneck due to their size and cannot be accommodated on a single less-powerful GPU. To address this challenge and assist the GPU-poor during this scarcity era, an ideal LLM serving framework should be universal and capable of deploying on a broader spectrum of devices, highly performant, and able to scale up with the addition of more GPUs.

Machine learning compilation (MLC) techniques offer such a solution, which allow **universal deployment** on all GPUs. Backed by TVM Unity, the latest generation of Apache TVM, MLC LLM is capable of generating highly performant code across all GPU backends, such as CUDA, ROCm, Vulkan, OpenCL, Metal, and WebGPU, achieving **state-of-the-art performance** across NVIDIA, AMD, and Intel GPUs. Most importantly, unlike a traditional C++ compiler, it compiles for both single-node and **multi-GPU and distributed** use cases, as machine learning necessitates.

In this blog post, MLC LLM demonstrates multiple advantages in serving large-scale LLMs:

- Performance: It compiles and generates highly performant code for multi-GPU inference.
- Universal deployment: It empowers universal deployment of LLMs onto GPUs across a variety of vendors and architectures.
- Scalability: It meaningfully scales and provides more acceleration as the number of GPUs increases.

## MLC-Powered Multi-GPU Inference

This section demonstrates MLC LLM’s multi-GPU capability by answering the following questions via experiments:

- Performance: How fast is the solution? Is MLC LLM sufficiently performant to serve large-scale LLMs?
- Scalability: How well can we scale? Will the latency meaningfully improve with more GPUs added to the system?
- Universal deployment: Which GPU backends does MLC LLM support out of the box?

### Settings

We focus on auto-regressive decoding performance in this blog post. Long-context prefilling is left for future blog series.

**GPUs**. We have chosen GPUs from two representative groups: server- and consumer-class GPUs. In the server class, we use A100 (80GB) and A10g (24GB), and in consumer-class GPUs, we use NVIDIA’s RTX 4090 (24GB) and AMD Radeon 7900 XTX (24GB). All numbers are based on PCIe, not NVLink.

**Models**. We focus on sufficiently large models; in our case, CodeLlama-34B and Llama2-70B. The 4-bit quantized model could fit into 2 GPUs in all our experimental settings, while FP16 numbers are provided whenever VRAM is sufficient.

**Drivers and OS**. All the experiments are run on Ubuntu 22.04. For NVIDIA GPUs, we use CUDA 12.1, except for vLLM, which only supports CUDA 11.8 at the moment. For AMD GPUs, we use ROCm 5.7, the latest generation of ROCm.

### Performance

<p align="center">
  <img src="/img/multi-gpu/figure-1.svg" width="80%">
  <figcaption>Figure 1. FP16 performance on two RTX 4090.</figcaption>
</p>

We first examine the single-batch decoding performance of the solution on two RTX 4090 GPUs. This configuration allows us to effectively work with Llama-70B using 4-bit setups. To provide a reference point, we also benchmarked other solutions in this setup. This post focuses on the optimal latency that a multi-GPU system could possibly achieve; the reference frameworks may not be optimized for a multi-GPU latency-focused scenario. Nevertheless, we include these reference results to help us gauge the performance of the multi-GPU solution.

On two RTX 4090 GPUs, we achieve 34 tokens/sec on Llama2-70B and 64 tokens/sec on CodeLlama-34B. It is remarkable to witness such a significant speedup with consumer GPUs, which are also accessible to those with limited access to high-end GPUs.

### Scalability

The second question we ask is how the solution scales with the number of GPUs. The figure on the left below shows the performance of FP16 on A100-PCIe and A10G-PCIe, while the figure on the right shows the performance of 4-bit-quantized models on the same set of server-class GPUs. FP16 serving numbers are not provided for Exllama because it does not support FP16. Across all the experiments, we can observe that our solution continues to improve performance as we increase the number of GPUs.

<p align="center">
  <img src="/img/multi-gpu/figure-2.svg" width="80%">
  <figcaption>Figure 2. Scaling MLC LLM across multiple A100 and A10G GPUs.</figcaption>
</p>

It is worth mentioning that the scaling is not yet linear in this case. This is because we are working on a more challenging strong scaling case while keeping the single batch for latency optimal settings. There are also other related factors that contribute to the result:

First, the absence of NVLink. We noticed that PCIe and the host CPU may significantly contribute to the overhead and potential noise in our 8xA100 experiment. More specifically, LLM performance could degrade by up to 30% when the CPU is busy with tasks from other tenants, indicating that a much faster independent data path could potentially help with performance.

Also, resource under-saturation. It is generally more challenging for auto-regressive decoding to saturate GPU resources. We noticed that the kernel speeds up less proportionally as workloads reduce on each GPU. For example, the multi-head attention kernel from cutlass operates at a similar latency with half of the attention heads in one of our experiments.

Nevertheless, having the ability to scale means we can achieve faster speeds or simply leverage multiple resource-constrained devices to serve even bigger models.

### Universal deployment

There have been many LLM inference solutions since the bloom of open-source LLMs. Most of the performant inference solutions are based on CUDA and optimized for NVIDIA GPUs. Meanwhile, due to the high demand for compute availability, it is useful to extend support to a broader class of hardware accelerators, with AMD being one potential candidate.

<p align="center">
  <img src="/img/multi-gpu/figure-3.svg" width="80%">
  <figcaption>Figure 3. Deploying MLC LLM on both NVIDIA RTX 4090 and AMD GPU Radeon 7900 XTX.</figcaption>
</p>

By adopting the universal deployment approach, MLC enables us to deploy on AMD GPUs through ROCm. We tested the same solution on two AMD 7900 XTX GPUs, and the results showed that these two AMD GPUs can achieve 30 tokens/sec for Llama2-70B. This indicates that we can achieve approximately 85% of the results produced by two RTX 4090 GPUs.

Considering that AMD GPUs cost $1000 per card, the setup with two AMD GPUs can be cost-effective for running Llama 70B models. This result suggests that, given the right price and availability, AMD GPUs could become valuable for LLM inference."

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
    --max-seq-len 2048 \
    --num-shards 2 \ # e.g. 2, 4, 8
    --target cuda --use-cuda-graph
```

**Step 3. Run Llama2-70B using Python API**. The following Python script showcases the Multi-GPUs inference of MLC LLM:

```python
from mlc_chat import ChatModule, ChatConfig
from mlc_chat.callback import StreamToStdout

cm = ChatModule(
  model="Llama-2-70b-chat-hf-q4f16_1",
  chat_config=ChatConfig(num_shards=2),
)
cm.generate(
    prompt="What is the meaning of life?",
    progress_callback=StreamToStdout(callback_interval=2),
)
```

## Discussion and Future works
**Machine Learning Compilation**. We leverage Apache TVM Unity, the latest machine learning compilation techniques that allow the representation and optimization of machine learning programs. MLC utilizes cross-layer representation and optimization for an end-to-end machine learning stack and makes use of lower-level compilers (LLVM/MLIR) and libraries to generate binary code. We model multi-GPU inference in TVM’s IRModule through a Single-Program-Multiple-Data (SPMD) representation. It further reduces collective library calls to NCCL/RCCL, highly optimized by NVIDIA and AMD. With MLC, we can conveniently represent pipelines, tensor parallelism, their lowering, and potential cross-layer optimization opportunities.

This post is part of the ongoing effort to bring high-performance universal deployment via MLC. We are also actively working on several areas to generalize our study:

- Enabling batching and speculation support;
- Integrating with the PyTorch ecosystem;
- Empowering more quantization and model architectures;
- Optimizations for long-context.

Our final takeaway is that machine learning system engineering is a continuous challenge. The key question is not only about building the right solution now but also about how to keep up and continuously bring ML engineering to new platforms. Productivity in machine learning engineering is crucial. Thanks to the Python-first ML compilation development flow, we have a universal multi-GPU solution and can integrate it with our past improvements in multi-backend code generation. We anticipate that related approaches will become even more valuable as we explore more ideas to achieve universal deployments and address the hardware availability problem.
