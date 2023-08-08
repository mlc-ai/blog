---
layout: post
title:  "CUDA-level Performance LLM Inference on AMD GPU"
date:   2023-08-08 09:30:00 -0400
author:   MLC Community
notitle: true
---

## TL;DR


MLC-LLM makes it possible to compile LLMs and deploy them on AMD GPUs using ROCm with **CUDA-level performance**. More specifically, AMD Radeon™ RX 7900 XTX gives 80%~85% of the speed of NVIDIA® GeForce RTX™ 4090 for Llama2-7B/13B.

<p align="center">
  <img src="/img/amd.svg" width="60%">
</p>

Our solution also generalizes to a variety of AMD GPUs and APUs. We also deployed Llama 2 7B on SteamDeck using Vulkan, 
which has an AMD APU. The speed is 13.2 tok/s.

Follow the instructions below 🔽 to try it out yourself if you have an AMD card!

## Background

There has been many LLM inference solutions since the bloom of open-source LLMs.
Most of the performant inference solutions are based on CUDA and optimized for nvidia GPUs.
In the meantime, with the high-demand for compute availability, it is useful to bring
support to a broader class of hardware accelerators. AMD is one potential candidate.

|                  | AMD Radeon™ RX 7900 XTX | NVIDIA ® GeForce RTX™ 4090 |
|:----------------:|:-----------------------:|:--------------------------:|
|       Cores      |  6144 stream processors |      16384 CUDA cores      |
|      Memory      |        24GB GDDR6       |         24GB GDDR6X        |
| Memory Bandwidth |         960 GB/s        |          1008 GB/s         |
|        TDP       |           320W          |            450W            |
|       Price      |           999$          |            1599$           |

In this post, we are taking deep look at the large language model inference problem
and see how well can AMD GPUs do. We specifically looks at single-batch 4-bit LLM 
inference problem as a starting point. We are also interested in asking how well 
can AMD GPUs do compared to a performant CUDA solution on Nvidia GPUs.


## Machine Learning Compilation

In order to peform such comparison. We will need a inference framework that universally deployed
and more importantly, optimizes for both Nvidia and AMD GPUs.

Here we leverage MLC-LLM, a inference framework that offers high-perfomance universal deployment for LLMs.
Specifically MLC-LLM bring state of art performance for a wide variety of backends, including CUDA, Metal,
Rocm, Vulkan, and OpenCL, spanning both server class GPUs to mobile(iPhone and Android).

At a high level, the framework let user take open language models and provide python-based API to
productively transform, and optimize the tensor computations in the model inference workload, 
and generates code for the platform of interest.

MLC-LLM leverages machine learning compilation, an emerging technology that compiles and automates 
optimization of machine learing programs. Specificially, we build solution on Apache TVM unity, a deep-learning compiler that utilizes a unified IR to represent the DL model at both graph and operator levels throughout the compilation stages. It allows *customizable model construction, composable model transformation, and transferable kernel optimization* for ML engineers to effectively customize and reuse ML compilation pipelines, reducing the effort of repeatedly implementing the same mechanism for different models or backends. TVM Unity also implements universal deployment runtime that enables developers to deploy the solution to programming language and platform of their choice.

What makes TVM Unity different and even more productive is the Python-first development flow, where we can

* inspect and modify the computational graph in Python
* compose IR transformations in Python
* inspect and write self-defined operators in Python, and compile them with other pre-defined operators composably in the same computational graph
* write the kernel optimization generically in Python and the compiler generates shader language codes for different backends accordingly, which allows us to transfer the kernel optimization techniques across backends

We are leveraging the python-first development, and universal deployment solution to quickly enable high-performance AMD GPU 
support less than one human week's effort.

### Bringing ROCm support to MLC 

There are several possible ways to support AMD GPU: RoCm, OpenCL, Vulkan, WebGPU.
ROCm stack is what AMD recently push for and have a lot of the corresponding 
building blocks similar to CUDA stack. 
Vulkan is the latest graphics standard and offers the most wide range of support
across GPU devices. WebGPU is the latest web standard that allows the compute to run on web browsers.

MLC can automatically generate code for all of them so we can also do some cross comparisons. 
We pick ROCm for most of our results in the 7900 txt and use vulkan(mesa driver) for steamdeck.

Our rocm support flow is as follows:

- Reuse the whole MLC pipeline for existing targets, including CUDA and Metal, which includes high-level optimizations
  such as static memory planning for dynamic computation and operator fusion, etc.
- We reused a generic GPU kernel optimization space written in TensorIR and do some profiling to specialize
  the hyper parameters for AMD cards. Importantly, this kernel transformation is purely written in python
  allowing us to do such optimization in the order of a day.
- We leverage ROCm LLVM backend to translate the IR of each kernel to ROCm code.
- Finally, everything is packed into a shared library that can be invoked by python and rest APIs.


## Benchmark

The models we are testing are Llama 2 7B and 13B with 4-bit quantization. And we measure the decoding performance by setting prompt tokens=1 and generate 512 tokens.

|                  | AMD Radeon™ RX 7900 XTX | NVIDIA ® GeForce RTX™ 4090 |
|:----------------:|:-----------------------:|:--------------------------:|
|        7B        |       134.3 tok/s       |         164.3 tok/s        |
|        13B       |        75.2 tok/s       |         94.4 tok/s         |

For single batch inference performance, it can reach 80%~85% of the speed of NVIDIA 4090 with the release of ROCm 5.6.

```python

```

NOTE: fold instructions into benchmark,  include python

We provide prebuilt wheels and instructions so you can also try these out on your own devices.


## Bringing Support to Broader Range of AMD Devices

After having fun with 7900 XTX. Let us start to look into a broader set of AMD devices.
Specifically, we looked into SteamDeck, which comes with a AMD APU.
One limitation of the deck is that the bios caped the GPU VRAM to 4GB, which is not
enough for rocm driver to support a 4-bit 7B model. Luckily, we find out that
Mesa's vulkan driver on steamdeck have robust support that allows buffer to go
beyond the 4GB cap(likely reuses some unified memory on CPU). 

```
(deck@steamdeck mlc-llm)$ ./build/mlc_chat_cli --local-id Llama-2-7b-chat-hf-q4f16_1
Use MLC config: "/home/deck/mlc-llm/dist/Llama-2-7b-chat-hf-q4f16_1/params/mlc-chat-config.json"
Use model weights: "/home/deck/mlc-llm/dist/Llama-2-7b-chat-hf-q4f16_1/params/ndarray-cache.json"
Use model library: "/home/deck/mlc-llm/dist/Llama-2-7b-chat-hf-q4f16_1/Llama-2-7b-chat-hf-q4f16_1-vulkan.so"
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

We applied the vulkan backend on this device, and successfully deployed Llama 2 7B 13.2 tok/s.
This results shed some lights on how a broad spectrum of AMD devices can be supported
for different consumers.

## Discussions and Future Works

With the arrival of generative AI, we are facing a hardware availability issue. 
The ability to bring a broad spectrum of hardware devices and make them performant is more important than ever.
In this post, we show that with the right software support, AMD GPU can get to CUDA-level performance
on large-language model inference for the latency sensitive use-cases. 

Although our study focuses on consumer grade GPUs, our past experience is that as we 
optimizes for 4090, we also observe correlated performance improvements on A10g and A100.
So we are confident that the study generalizes to server-grade GPUs and will update our study
once we have access to those devices.
With the set of evidences so far, we believe that with the right price and availability, 
AMD GPUs can start be effective for LLM inference.

This post is part of ongoing effort on bringing high-performance universal deployment via MLC. 
We are also actively working on several areas that can generalize our study.
- Enable batching and multiGPU support.
- Bringing connections to the PyTorch ecosystem.
- Enabling more quantization and model architectures.
- Enabling more automatic hardware backend optimizations.

You are more than welcomed to checkout to learn more

## Acknowledgement

The overall MLC projects are only possible thanks to the shoulders of open-source ecosystems that we stand on. We would love to continue developing and supporting the open-source ML community. We want to thank the Apache TVM community and developers of the TVM Unity compiler. The open-source ML community members made these models publicly available. PyTorch and Hugging Face communities make these models accessible. We would like to thank the teams behind RedPajama, Dolly, Vicuna, SentencePiece, LLaMA, and Alpaca. We also would like to thank OpenCL, Vulkan, C++, Python, and Rust communities that enable this project.
