---
layout: post
title:  "CUDA-level Performance LLM Inference on AMD GPU"
date:   2023-08-08 09:30:00 -0400
author:   MLC Community
notitle: true
---

## TL;DR

MLC-LLM makes it possible to compile LLMs and deploy them on AMD GPUs using **ROCm** with CUDA-level performance. More specifically, AMD Radeon™ RX 7900 XTX gives **80%** of the speed of NVIDIA® GeForce RTX™ 4090 and **94%** of the speed of NVIDIA® GeForce RTX™ 3090Ti for Llama2-7B/13B. Besides ROCm, our Vulkan support allows us to generalize LLM deployment to other AMD devices, for example, a SteamDeck with an AMD APU.

<p align="center">
  <img src="/img/amd/amd.svg" width="70%">
</p>

## Background

There have been many LLM inference solutions since the bloom of open-source LLMs.
Most of the performant inference solutions are based on CUDA and optimized for Nvidia GPUs.
In the meantime, with the high demand for compute availability, it is useful to bring
support to a broader class of hardware accelerators. AMD is one potential candidate.

<p align="center">
  <img src="/img/amd/7900-4090.png" width="90%">
</p>

<!-- |                  | AMD Radeon™ RX 7900 XTX | NVIDIA ® GeForce RTX™ 4090 | NVIDIA ® GeForce RTX™ 3090 Ti |
|:----------------:|:-----------------------:|:--------------------------:|:-----------------------------:|
|       Cores      |  6144 stream processors |      16384 CUDA cores      |        10752 CUDA cores       |
|      Memory      |        24GB GDDR6       |         24GB GDDR6X        |          24GB GDDR6X          |
| Memory Bandwidth |         960 GB/s        |          1008 GB/s         |           1008 GB/s           |
|        TDP       |           320W          |            450W            |              450W             |
|       Price      |           999$          |            1599$           |             1999$             | -->

### Discussions on HW spec
From the spec comparison, we can see that AMD 7900 XTX is a good match for Nvidia 4090 and 3090 Ti.
* All have 24GB memory, which means they can fit models of the same size.
* All have similar memory bandwidth, considering LLM inference is largely memory bound, 
  we can expect similar performance.
* Most importantly, AMD 7900 XTX is 40% (50%) cheaper than Nvidia 4090 (3090 Ti). So the performance (toks/sec) per dollar can be much better if we can get a similar performance.

In this post, we are taking a deep look at how well AMD GPUs can do compared to a performant CUDA solution on Nvidia GPUs.

## Machine Learning Compilation for ROCm

**What is machine learning compilation (MLC).** MLC is an emerging technology that compiles and automates optimization of machine learning workloads. Here we leverage MLC-LLM, an ML compilation-based solution that offers **high-performance universal deployment** for LLMs. Specifically, MLC-LLM brings state-of-the-art performance for a wide variety of backends, including CUDA, Metal, ROCm, Vulkan, and OpenCL, spanning both server-class GPUs to mobile (iPhone and Android). At a high level, the framework lets the user take open language models and compiles it with python-based workflow, including APIs to transform computational graphs, optimize layout and scheduling of GPU kernels, and deploys it natively on platforms of interest.

<!-- MLC-LLM leverages machine learning compilation, an emerging technology that compiles and automates 
optimization of machine learning programs. Specifically, we build a solution on Apache TVM unity, a deep-learning compiler that utilizes a unified IR to represent the DL model at both graph and operator levels throughout the compilation stages. It allows *customizable model construction, composable model transformation, and transferable kernel optimization* for ML engineers to effectively customize and reuse ML compilation pipelines, reducing the effort of repeatedly implementing the same mechanism for different models or backends. TVM Unity also implements universal deployment runtime that enables developers to deploy the solution to the programming language and platform of their choice.

What makes TVM Unity different and even more productive is the Python-first development flow, where we can

* Inspect and modify the computational graph in Python
* Compose IR transformations in Python
* Inspect and write self-defined operators in Python, and compile them with other pre-defined operators composable in the same computational graph
* Write the kernel optimization generically in Python and the compiler generates shader language codes for different backends accordingly, which allows us to transfer the kernel optimization techniques across backends

We are leveraging the Python-first development, and universal deployment solution to quickly enable high-performance AMD GPU 
support less than one human week's effort. -->

<p align="center">
  <img src="/img/amd/arch.svg" width="80%">
</p>

**MLC for AMD GPUs and APUs.** There are several possible ways to support AMD GPU: ROCm, OpenCL, Vulkan, and WebGPU. ROCm stack is what AMD recently push for and has a lot of the corresponding 
building blocks similar to the CUDA stack. Vulkan is the latest graphics standard and offers the widest range of support across GPU devices. WebGPU is the latest web standard that allows the computation to run on web browsers. MLC supports automatic code generation targeting all the backends above.

We pick ROCm for Radeon 7900 XTX and Vulkan for Steamdeck's APU. We find that ROCm stack just works out of box and a few more hours to further bring an optimized version, thanks to the productive python development pipeline in MLC. We made the following things to use ROCm support from MLC:

- Reuse the whole MLC pipeline for existing targets (such as CUDA and Metal), including memory planning, operator fusion, etc.
- Reuse a generic GPU kernel optimization space written in TVM TensorIR and re-target it to AMD GPUs.
- Reuse TVM's ROCm code generation flow that generates low-level ROCm kernels through LLVM.
- Finally, export generated code as a shared or static library that can be invoked by CLI, Python and REST APIs.

## Benchmark with MLC Python Package

The models we are testing are Llama 2 7B and 13B with 4-bit quantization. And we measure the decoding performance by setting prompt tokens=1 and generating 512 tokens.

<p align="center">
  <img src="/img/amd/perf.png" width="60%">
</p>

<!-- |                  | AMD Radeon™ RX 7900 XTX | NVIDIA ® GeForce RTX™ 4090 |
|:----------------:|:-----------------------:|:--------------------------:|
|        7B        |       134.3 tok/s       |         164.3 tok/s        |
|        13B       |        75.2 tok/s       |         94.4 tok/s         | -->

For single batch inference performance, it can reach 80%~85% of the speed of NVIDIA 4090 with the release of ROCm 5.6.

### Try it out yourself!

We provide prebuilt wheels and instructions so you can also try these out on your own devices.

- Prerequisites: AMD GPU, with ROCm 5.6 or Vulkan support.
- Install mlc python packages: see instructions  https://mlc.ai/package/

if you are using ROCm on Linux, the installation command is
  
```bash
pip install --pre --force-reinstall mlc-ai-nightly-rocm mlc-chat-nightly-rocm -f https://mlc.ai/wheels

# Verify the installation of the Python package.
# You are expected to see "<class 'mlc_chat.chat_module.ChatModule'>" printed out.
python -c "from mlc_chat import ChatModule; print(ChatModule)"
```

- Download the quanzized model parameters and compiled model library

```bash
# Install Git and Git-LFS if you haven't already. Then run
git lfs install
mkdir -p dist/prebuilt

# compiled model library
git clone https://github.com/mlc-ai/binary-mlc-llm-libs.git dist/prebuilt/lib
cd dist/prebuilt
# quanzized model parameters
git clone https://huggingface.co/mlc-ai/mlc-chat-Llama-2-7b-chat-hf-q4f16_1; cd ../../
```

- Then test the perfomance with the following Python script, note that it should be put under the same folder with `dist` folder.

```python
from mlc_chat import ChatModule

# From the mlc-llm directory, run
# $ python examples/python/benchmark.py

# Create a ChatModule instance
cm = ChatModule(model="Llama-2-7b-chat-hf-q4f16_1")

output = cm.benchmark_generate("Hi", generate_length=512)
print(f"Generated text:\n{output}\n")
print(f"Statistics: {cm.stats()}")

# Reset the chat module by
# cm.reset_chat()
```

## Running on SteamDeck using Vulkan

After having fun with the 7900 XTX. Let us start to look into a broader set of AMD devices.
Specifically, we looked into SteamDeck, which comes with an AMD APU.
One limitation of the deck is that the bios caped the GPU VRAM to 4GB, which is not
enough for the ROCm driver to support a 4-bit 7B model. Luckily, we find out that
Mesa's Vulkan driver on Steamdeck has robust support that allows the buffer to go
beyond the 4GB cap (likely reuses some unified memory on the CPU). 

```
(deck@steamdeck mlc-llm)$ ./build/mlc_chat_cli --local-id Llama-2-7b-chat-hf-q4f16_1
Use MLC config: "/home/deck/mlc-llm/dist/Llama-2-7b-chat-hf-q4f16_1/params/mlc-chat-config.json"
Use model weights: "/home/deck/mlc-llm/dist/Llama-2-7b-chat-hf-q4f16_1/params/ndarray-cache.json"
Use model library: "/home/deck/mlc-llm/dist/Llama-2-7b-chat-hf-q4f16_1/Llama-2-7b-chat-hf-q4f16_1-Vulkan.so"
You can use the following special commands:
  /help               print the special commands
  /exit               quit the cli
  /stats              print out the latest stats (token/sec)
  /reset              restart a fresh chat
  /reload [local_id]  reload model `local_id` from disk, or reload the current model if `local_id` is not specified

Loading model...
Loading finished
Running system prompts...
System prompts finished
[INST]: Hi
[/INST]: Hello! I'm here to help you with any questions or concerns you may have. However, I must inform you that I cannot provide advice or suggestions that promote or facilitate harmful or illegal activities. It is important to always act in a safe and responsible manner, and to respect the laws and well-being of yourself and others. Is there anything else I can help you with?
[INST]: /stats
prefill: 48.3 tok/s, decode: 13.2 tok/s
```

We applied the Vulkan backend on this device, and successfully deployed Llama 2 7B 13.2 tok/s.
These results shed some light on how a broad spectrum of AMD devices can be supported
for different consumers.

## Discussions and Future Works

With the arrival of generative AI, we are facing a hardware availability issue. 
The ability to bring a broad spectrum of hardware devices and make them performant is more important than ever.
In this post, we show that with the right software support, AMD GPU can get to CUDA-level performance
on large-language model inference for the latency sensitive use cases. 

Although our study focuses on consumer-grade GPUs, our experience is that as we 
optimize for 4090, we also observe correlated performance improvements on A10g and A100.
So we are confident that the study generalizes to server-grade GPUs and will update our study
once we have access to those devices.
With the set of evidence so far, we believe that with the right price and availability, 
AMD GPUs can start to be effective for LLM inference.

This post is part of an ongoing effort on bringing high-performance universal deployment via MLC. 
We are also actively working on several areas that can generalize our study.
- Enable batching and multi-GPU support.
- Bringing connections to the PyTorch ecosystem.
- Enabling more quantization and model architectures.
- Enabling more automatic hardware backend optimizations.

You are more than welcome to check out to learn more

## Links

Please refer to our [project page](https://mlc.ai/mlc-llm/) for a detailed guide on how to try out the MLC LLM deployment. The source code of MLC LLM is available on our official [GitHub repository](https://github.com/mlc-ai/mlc-llm/). You are also more than welcomed to join the [Discord Channel](https://discord.com/invite/9Xpy2HGBuD) for further discussion.

## Acknowledgement

The overall MLC projects are only possible thanks to the shoulders of open-source ecosystems that we stand on. We would love to continue developing and supporting the open-source ML community. We want to thank the Apache TVM community and developers of the TVM Unity compiler. The open-source ML community members made these models publicly available. PyTorch and Hugging Face communities make these models accessible. We would like to thank the teams behind RedPajama, Dolly, Vicuna, SentencePiece, LLaMA, and Alpaca. We also would like to thank OpenCL, Vulkan, C++, Python, and Rust communities that enable this project.
