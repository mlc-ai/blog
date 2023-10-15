---
layout: post
title:  "可拓展的 NVIDIA/AMD 多卡大语言模型部署"
date:   2023-10-13 09:30:00 -0400
author:   MLC Community
notitle: true
---

* TOC
{:toc}

## TL;DR

机器学习编译技术 (Machine Learning Compilation, MLC) 使得在多张 NVIDIA 或 AMD 显卡上高性能部署大规模语言模型成为可能。具体而言，运用 MLC 技术，4-bit 量化的 Llama2-70B 可在两张 NVIDIA RTX 4090 上以 34.5 tok/sec 的速度运行，在两张 AMD Radeon 7900XTX 上以 29.9 tok/sec 的速度运行。此外，我们还观察到随着显卡数量的增加，MLC 始终保持良好的可拓展性，在八张 A100/A10G 的机器上做单批次推断，依然能做到随着显卡增多，延迟不断改善。

## 背景

显卡短缺，作为近年来困扰全球的问题，已然导致它们的价格不断攀升、供应越发有限。尽管如此，学术界和工业界对大规模大语言模型 (LLMs) 的需求依旧强烈。然而，许多最大的模型，如 Meta 的 Llama2-70B，由于其模型大小而面临严重瓶颈，无法在单个普通显卡上运行。为了解决这一挑战，让普通显卡也能跑起大规模大语言模型，一个理想的 LLM 服务框架应当足够通用，能在更广泛的设备上高性能部署，并且能够随着添加更多显卡而不断提升性能。

机器学习编译 (MLC) 技术提供了这样一个解决方案，它允许在所有显卡上进行通用部署。在 Apache TVM 的最新一代 TVM Unity 支持下，MLC LLM 能够在**所有显卡后端** (如 CUDA、ROCm、Vulkan、OpenCL、Metal 和 WebGPU) 上生成高性能代码，在 NVIDIA、AMD 和 Intel 显卡上实现**最佳的性能**。最重要的是，与传统的 C++ 编译器不同，它既可以为单节点，也可以为**多显卡和分布式**使用情况进行编译，从而支持机器学习算法的需求。

通过这篇博文，我们展示了 MLC LLM 在部署大规模大语言模型时的多个优势：

- 性能：它能为多显卡分布式推断生成高性能代码。
- 通用部署：它使得大规模大语言模型能在各个厂商的各种显卡后段和架构上通用部署。
- 可拓展性：它的分布式推理方案具有显著的拓展性，随着显卡数量的增加，提供更多的加速。

## 基于机器学习编译的多卡推理

本节通过实验回答以下问题，进而展示 MLC LLM 的多卡部署能力：

- 性能：它是否足够高性能以服务大规模大语言模型？
- 可拓展性：是否能通过增加更多显卡不断改善延迟？
- 通用部署：它的分布式推理默认支持哪些 GPU 后端？

### 实验设定

在这篇博文中，我们关注自回归解码性能。长上下文填充留待将来的博文系列来讨论。

**显卡**. 我们选择了来自两个代表性群体的显卡：服务器级别和消费级别的GPU。在服务器级别，我们使用 A100 (80GB) 和 A10G (24GB)，在消费级别的GPU中，我们使用NVIDIA的RTX 4090（24GB）和AMD Radeon 7900 XTX（24GB）。所有显卡通信均基于 PCIe 而非 NVLink。

**模型**. 我们关注足够大的模型；在我们的实验设置中，我们使用 CodeLlama-34B 和 Llama2-70B。在所有实验中，4-bit 量化模型都可以放在双显卡上，而当显存足够时，我们提供 FP16 的数字。

**驱动程序和操作系统**. 所有实验都在 Ubuntu 22.04 上运行。对于 NVIDIA 显卡，除了vLLM 目前只支持 CUDA 11.8 以外，我们使用均 CUDA 12.1。对于 AMD 显卡，我们使用最新一代的 ROCm 5.7。

### 性能

我们首先在两块 RTX 4090 显卡上实验了 MLC LLM 的单批次解码性能。这个配置使我们能够有效地使用 4-bit Llama-70B。同时，为了提供一些参考基准，我们也对其他各种解决方案进行了测试。注意到这些参考框架可能并非均为关注延迟的多卡推理而设计优化，因此结果仅有一定的参考意义。

<p align="center">
  <img src="/img/multi-gpu/figure-1.svg" width="50%">
  <figcaption>Figure 1. 4-bit 量化的 CodeLlama-34B 和 Llama2-70B 在两个 NVIDIA RTX 4090 上的单批次性能.</figcaption>
</p>

在两块 RTX 4090 显卡上，我们在 Llama2-70B 上达到了 34 tok/秒，在 CodeLlama-34B 上达到了 64 tok/秒的性能。值得注意的是，我们的解决方案在消费级显卡中达到显著的性能，为无法访问旗舰显卡的厂商提供了一个切实可行的大语言模型推断方案。

### 可拓展性

我们问的第二个问题是解决方案随着显卡数量的增加而如何扩展。对此，我们同样也做了两组实验，左下图显示了 A100-80G-PCIe 和 A10G-24G-PCIe 上的 fp16 性能，而右下图显示了同一组云端显卡上 4-bit 量化后的模型性能。我们没有提供 Exllama fp16 的数字因为它没有提供支持。在所有实验中，我们都可以观察到随着显卡数量增加，我们的解决方案的延迟不断降低。

<p align="center">
  <img src="/img/multi-gpu/figure-2.svg" width="100%">
  <figcaption>Figure 2. fp16 和 4-bit 量化后的 CodeLlama-34 和 Llama2-70B 在 8 块 A100-80G-PCIe 和 8 块 A10G-24G-PCIe 上的拓展性.</figcaption>
</p>

值得一提的是，在这种设定下，我们依然没有做到线性拓展。这是因为我们的实验设定是优先单批次低延迟情境，为了拿到最低延迟，会牺牲一些拓展性。此外，还有一些其他相关因素影响了结果：首先，我们的设备没有搭载 NVLink，因此所有通信均经过 PCIe 和 CPU。我们注意到这可能会显著增加我们 8 块 A100 实验中的额外开销和潜在噪声。举个例子，在一次实验中，当 CPU 在忙于处理和我们共用机器的其他租户的请求时，推理性能可能会下降高达 30%。这表明一个更快的独立数据通路可能有助于提高性能。此外，显卡资源没有饱和利用。对于单批次自回归解码来说，打满显卡资源通常非常具有挑战性。我们注意到，随着每个显卡上的工作负载减少，每个算子的速度并没有成比例的增加。例如，在我们的一个实验中，当注意力头的数量减少一半时，cutlass 的多头注意力的延迟并没有下降。

尽管如此，具有良好的可拓展性意味着我们可以通过简单增加显卡的方式实现更低的延迟，或者简单地利用多个资源受限设备来服务更大的模型。

### 通用部署

自从开源大模型兴起以来，就有许多大模型推断的解决方案。大多数性能出色的推断解决方案都基于 CUDA 并且针对 NVIDIA GPU 进行了重点优化。
与此同时，由于计算资源需求大，支持除 CUDA 以外的计算平台和加速器的收益也很明显。其中，AMD 显卡是一个潜在的选择。

<p align="center">
  <img src="/img/multi-gpu/figure-3.svg" width="30%">
  <figcaption>Figure 3. 4-bit 量化的 Llama2-70B 和 CodeLlama-34B 在两块 GPU 单批次推理性能：NVIDIA RTX 4090 与 AMD Radeon 7900 XTX.</figcaption>
</p>

TVM Unity 的通用部署能力使得 MLC LLM 能够利用其 ROCm 后段在 AMD 显卡上部署。据此，我们在两块 AMD 7900 XTX 显卡上测试了相同的解决方案，结果显示这两块 AMD 显卡可以在 Llama2-70B 上达到 30 tok/sec，也就是 NVIDIA RTX 4090 显卡 85% 左右的性能。考虑到 AMD 显卡的价格为每张卡 1000 美元，使用两个 AMD 显卡的设置可能是运行 Llama2-70B 模型最经济有效的方法之一。因此，运用 MLC LLM 的通用部署能力，我们能够将 AMD 显卡变成经济实惠且性能强劲的大模型推断方案。

## 使用 MLC LLM

### Docker

我们在一个专用的 GitHub [仓库](https://github.com/mlc-ai/llm-perf-bench) 存放了 MLC LLM 的相关使用说明。该仓库允许我们复现 MLC LLM 在单卡、多卡和 CUDA/ROCm 上的性能。

### Python API

下面我们展示一下如何在 纯Python 中使用多卡推断。

**步骤 0. 安装 MLC LLM Python 包**. 创建一个隔离的 conda 虚拟环境，然后通过[官网提供的指令](https://llm.mlc.ai/docs/install/mlc_llm.html)安装 MLC LLM nightly 版本。以下是详细指令：

```bash
VENV=mlc-llm-venv
conda create -n $VENV -c conda-forge python numpy pytorch-cpu scipy
conda activate $VENV
pip install .... # 指令: https://llm.mlc.ai/docs/install/mlc_llm.html
```

**步骤 1. 下载量化后的模型参数**. 按照下面的命令从 HuggingFace 下载 4-bit 量化的 Llama2-70B:

```bash
MODEL=Llama-2-70b-chat-hf

git lfs install
git clone https://huggingface.co/mlc-ai/mlc-chat-$MODEL-q4f16_1 ./dist/$MODEL-q4f16_1/params
```

**步骤 2. 编译多卡推断**. MLC LLM 的编译器由纯 Python 实现。下面的指令允许我们下载 MLC LLM 的编译器，然后使用它来编译模型。这可能需要几秒钟的时间。调整 “--num-shards” 以设置要使用的显卡数量：

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

**步骤 3. 使用 MLC LLM 的 Python API 运行 Llama2-70B**. 使用下面的 Python 脚本使用 MLC LLM 的多卡推理功能：

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

## 更多讨论，未来的工作

**机器学习编译**. MLC LLM 项目广泛使用 Apache TVM Unity，这是一种允许表示和优化机器学习程序的最新机器学习编译技术。MLC 注重于跨层的联合表示和联合优化，并通过低层级的编译器（LLVM/MLIR）和库来生成二进制代码。在多卡推理中，我们即利用 TVM 的 Single-Program-Mulit-Data (SPMD) 表示对多显卡推断进行建模。这个 SPMD 表示会进一步下降到由 NVIDIA 和 AMD 高度优化的 NCCL/RCCL 库中，并且基于 TVM，我们可以方便地表示和发现管道、张量并行、它们的下降以及潜在的跨层优化机会。

本文是通过MLC带来高性能通用部署的持续努力的一部分。我们还在积极开展以下几个方面的研究：

- 支持 batching 和 speculative decoding；
- 与 PyTorch 生态系统集成；
- 支持更多的量化和模型架构；
- 长上下文的优化。

我们最后的结论是，机器学习系统工程是一个持续的挑战。关键问题不仅仅是构建一个``当下可行''的解决方案，还包括如何引进最新的机器学习研究和工程，支持最新的平台。提高生产力在机器学习工程中至关重要：通过 Python 优先的机器学习编译开发流程，TVM 编译器使得迅速搭建一个多卡推理方案成为可能。我们预计，随着我们在通用部署方案的探索，相关的方法将变得更有价值。