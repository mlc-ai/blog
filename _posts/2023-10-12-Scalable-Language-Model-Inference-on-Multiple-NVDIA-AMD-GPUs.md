---
layout: post
title:  "Scalable Language Model Inference on Multiple NVDIA/AMD GPUs"
date:   2023-10-12 09:30:00 -0400
author:   MLC Community
notitle: true
---

* TOC
{:toc}

## TL;DR

Machine Learning Compilation (MLC) now supports deploying gigantic language models on multiple GPUs. It runs 4bit-quantized Llama2-70B at 34.5 tok/sec on two NVIDIA RTX 4090 and 29.9 tok/sec on two AMD Radeon 7900XTX. MLC techniques consistently scale and accelerate with more GPUs in our experiments.

## Background

We are currently facing a global shortage of powerful GPUs, resulting in their increasing cost and limited availability. Despite this, the demand for gigantic Large Language Models (LLMs) remains strong within academia and industry. However, many of the largest models, such as Meta's Llama2-70B, face a bottleneck due to their size and cannot be accommodated on a single less-powerful GPU. To address this challenge and realistically assist the GPU poor during this scarcity era, an ideal LLM serving framework should be universal and capable of deploying on broader spectrum of devices, highly performant and scale up with the addition of more GPUs.

Machine learning compilation (MLC) techniques offer such a solution - it allows **universal deployment** on all GPUs. Backed by TVM Unity, the latest generation of Apache TVM, MLC LLM is capable of generating highly performant code across all GPU backends, such as CUDA, ROCm, Vulkan, OpenCL, Metal and WebGPU, achieving **state-of-the-art performance** across NVIDIA, AMD and Intel GPUs. Most importantly, unlike a traditional C++ compiler, it compiles for both single node and **multi-GPU and distributed** use cases as machine learning necessitates.

In this blog post, MLC LLM demonstrates multiple advantages on serving gigantic LLMs:

- **Performance:** It compiles and generates highly performant code for multi-GPU inference;
- **Universal deployment:** It empowers universal deployment of LLMs onto GPUs across a variety of vendors and architectures;
- **Scalability:** It meaningfully scales and brings more acceleration with number of GPUs increases.

## MLC-Powered Multi-GPU Inference

This section demonstrates MLC LLM’s multi-GPU capability by answering the following questions via experiments:

- What GPU models does MLC LLM support out of box?
- Is MLC LLM sufficiently performant to make poor’s GPUs economically advantageous?
- With more GPUs added into the system, will the latency meaningfully improve?

### Settings

We focus on auto-regressive decoding performance on this blog post. Long-context prefilling is left for future blog series.

**GPUs**. We choose GPUs from two representative groups: server- and consumer-class GPUs. In server class, we use A100 (80GB) and A10g (24GB), and in consumer class GPUs, we use NVIDIA’s RTX 4090 (24GB) and AMD Radeon 7900 XTX (24GB). All numbers are based on PCIe not NVLink.

**Models**. We focus on sufficiently large models, in our case, CodeLlama-34B and Llama2-70B. 4bit-quantized model could fit into 2 GPUs in all our experiment settings, while fp16 numbers are provided whenever VRAM is sufficient.

**Drivers and OS**. All the experiments are run on Ubuntu 22.04. For NVIDIA GPUs, we use CUDA 12.1, except for vLLM which only supports CUDA 11.8 at the moment; For AMD GPUs, we use ROCm 5.7, the latest generation of ROCm.

### Results

**FP16 on serving-class GPUs**. Figure 1 shows the performance of fp16 on server-class GPUs, A100-PCIe and A10G-PCIe. Exllama numbers are not provided because it does not support fp16.

<p align="center">
  <img src="/img/multi-gpu/figure-1.svg" width="80%">
  <figcaption>Figure 1. FP16 performance on A100-PCIe and A10G-PCIe.</figcaption>
</p>

**4-bit on server-class GPUs**. Figure 2 shows the performance of 4-bit-quantized models on the same set of server-class GPUs. Generally speaking, 4-bit models yield better results in most frameworks, as they are more memory bandwidth-friendly.

<p align="center">
  <img src="/img/multi-gpu/figure-2.svg" width="80%">
  <figcaption>Figure 2. 4-bit performance on A100-PCIe and A10G-PCIe.</figcaption>
</p>

**4-bit on consumer GPUs**. We managed to gain access to two NVIDIA RTX 4090 and two AMD Radeon 7900 XTX, and our results are presented in Figure 3.

<p align="center">
  <img src="/img/multi-gpu/figure-3.svg" width="80%">
  <figcaption>Figure 3. 4-bit performance on NVIDIA RTX 4090 and Radeon 7900 XTX.</figcaption>
</p>

### Analysis

As demonstrated in the experiments above, MLC LLM differentiates itself in all the multi-GPU settings. To answer the questions in the beginning of this section:

*What GPU models does MLC LLM support out of box?*

**Universal**. Preliminary experiments have shown that MLC LLM works out of box for both NVIDIA and AMD GPUs in multi-GPU setting, in both cloud and local settings for both fp16 and 4bit cases. With TVM Unity’s runtime, MLC LLM could also support Intel GPUs. Moreover, any performance improvement techniques for one backend can be almost transparently transferred to other backends without much effort.

*Is MLC LLM sufficiently performant to make poor’s GPUs economically advantageous?*

**Performant**. MLC LLM consistently outperforms the second-best solution by a substantial margin. It outperforms vLLM in fp16 serving by up to 2.7 times, and exceeds exllama in 4bit serving by up to 3.5 times. ML compilation techniques generate efficient GPU kernels that better utilize memory bandwidth and compute resources, and therefore, making MLC LLM a compelling choice for enhanced computational efficiency.

*With more GPUs added into the system, will the latency meaningfully improve?*

**Scalable**. Throughout our experiments, inference latency meaningfully reduces with more GPUs, meaning better latency could usually be achieved with more GPU resources. As a concrete example, two AMD Radeon 7900 XTX at the total cost of $2k could be 20% more efficient as 1 single A100 of more than $10k when serving int4 Llama2-70B.

## Discussion

### Scaling and Bandwidth

MLC LLM scales competitively across all of our experiments. In the ideal case, such as large batch distributed training, the scaling could be linear and proportional to the number of GPUs used. However, in single batch LLM inference, it is generally challenging to achieve the same level of scalability for the following reasons:

First, absence of NVLink. We noticed that PCIe and host CPU may contribute significantly to the overhead and potentially noise in our 8-A100-PCIe experiment. More specifically, LLM performance could degrade by up to 30% when CPU is busy with other tasks from other tenants, which indicating a much faster independent data path could potentially help with performance.

Also, resource under-saturation. It is generally more challenging for auto-regressive decoding to saturate GPU resources. We notice that the kernel speeds up only less proportionally as workloads reduce on each GPU. For example, the cutlass multi-head attention kernel operators at similar latency when we half the number of attention heads in one of our experiments.

### Server vs consumer-class GPUs

According to our experiments, 4bit-quantized Llama2-70B can be run smoothly with either cloud GPUs like A100, or NVIDIA and AMD’s consumer-class GPUs. In fact, there are two major noteworthy differences:

First is pricing. Two AMD Radeon 7900 XTX at the total cost of $2k could be 75% efficient as two A100s of more than $20k, but available at major cloud provider at hourly rate. Therefore, either solution more economical in certain scenarios.

Meanwhile, the optimization strategies could differ even if using same CUDA/HIP as language or the same LLVM IR as the low-level compiler. Cloud GPUs like A100 are usually equipped with faster TensorCore, which is helpful in large batch inference or training; whereas consumer GPUs are more powerful in CUDA core and higher clocks. It means on cloud GPUs, we will have to optimize more carefully to ensure memory bandwidth is fully saturated.

### Trading off latency and throughput

This blog post focuses on the optimal latency a multi-GPU serving system could possibly achieve with single batch inference. Other optimization techniques, for example, continuous batching and paged attention introduced by vLLM, allow processing of multiple requests concurrently, which primarily benefits throughput. Tweaking maximum allowed batch size helps to balance between latency and throughput - when batch size is 1, the optimal latency is achieved, while increasing batch size significantly helps throughput but may hurt latency.

## Instructions

### Docker

The Dockerfile and corresponding instructions are provided in a dedicated GitHub [repo](https://github.com/mlc-ai/llm-perf-bench) to reproduce MLC LLM performance for both single-GPU and multi-GPU, CUDA and ROCm.

### Python API

The instructions below showcase how to use the multi-GPU feature in pure Python.

**Install MLC LLM Python package**. Create an isolated conda virtual environment:

```bash
VENV=mlc-llm-venv
conda create -n $VENV -c conda-forge python numpy pytorch-cpu scipy
conda activate $VENV
```

And then install a nightly built of MLC LLM via pip following the [instructions](https://llm.mlc.ai/docs/install/mlc_llm.html):

```bash
pip install .... # Instructions: https://llm.mlc.ai/docs/install/mlc_llm.html
```

**Download pre-quantized weights**. The commands below download 4bit-quantized Llama2-70B from HuggingFace.

```bash
MODEL=Llama-2-70b-chat-hf

git lfs install
git clone https://huggingface.co/mlc-ai/mlc-chat-$MODEL-q4f16_1 ./dist/$MODEL-q4f16_1/params
curl https://raw.githubusercontent.com/mlc-ai/llm-perf-bench/main/model_configs/Llama-2-70b-chat-hf.json --create-dirs -o ./dist/models/$MODEL/config.json
```

**Compile for multi-GPU inference**. Download MLC LLM’s Python-based compiler:

```bash
git clone --recursive https://github.com/mlc-ai/mlc-llm && export PYTHONPATH=$(pwd)/mlc-llm/:$PYTHONPATH
```

Tweak "--num-shards" to set the number of GPUs to use, and then run compilation in command line. This could take a few seconds.

```bash
python3 -m mlc_llm.build --build-model-only \
    --model ./dist/models/$MODEL/ \
    --quantization q4f16_1 \
    --max-seq-len 2048 \
    --num-shards 2 \ # e.g. 2, 4, 8
    --target cuda --use-cuda-graph
```

**Run in Python**. The following Python script showcases the Multi-GPUs inference of MLC LLM:

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

## Machine Learning Compilation

Apache TVM Unity is the latest ML compilation techniques that allows representation and optimization of machine learning programs. Different from traditional source-in-binary-out compilation, MLC usually involves cross-layer representation and optimization for an end-to-end machine learning stack, and usually rely on lower-level compilers like LLVM/MLIR to generate binary code.

Multi-GPU inference, as a signature example of ML compilation, is modeled in TVM Unity in Single-Program-Multiple-Data (SPMD) representation. It further lowers to vendor library calls to NCCL/RCCL as highly optimized by NVIDIA and AMD. With MLC, we could conveniently represent pipeline, tensor parallelism, their lowering and potential cross-layer optimization opportunities.

Besides multi-GPU optimization in this blog post, ML compilation techniques will be powering more practical scenarios in LLM serving and deployment in the incoming months, such as batching, speculation and long context optimization.