---
layout: post
title:  Bringing Hardware Accelerated Language Models to Android Devices
date:   2023-05-08 23:00:00 -0400
author:   MLC Community
notitle: true
---

In this post, we introduce MLC LLM for Android -- a solution that allows large language models
to be deployed natively on Android devices, plus a productive framework for everyone to
further optimize model performance for their use cases.
Everything runs locally and accelerated with native GPU on the phone.


<p align="center">
  <img src="/blog/img/android/android-recording.gif" height="700">
</p>

We have witnessed significant progress in the field of generative AI and large language models. Thanks to the open source movement, we have witnessed the blooming moment of open source foundational models. While it is helpful to run those models on the server platforms. There are also great deal of potential to enable large-language models on consumer devices

<p align="center">
  <img src="/blog/img/android/local-advantage.png" width="90%">
</p>

Empowering LLMs on mobile devices is very important given this is what we interact with daily. In our announcement last week, we showed that we can bring LLMs to desktop and iOS devices.

Android is an operating system inside 2.5 billion active devices, and it is very interesting to ask if we can do the same with android devices as well, a lot of community members made the suggestion to support android devices. We agree, this post discusses how we achieve that goal in one week.

## How

Bringing hardware accelerated LLM models onto android devices is not as straightforward via traditional approaches, specifically, the two ecosystems have different sets of programming models both for host app programming and GPU programming, to name a few
 - We need to use different GPU programming models to provide hardware acceleration.
 - We need to optimize for different kinds of hardware backends. iOS devices are powered by Apple’s A16 chips, while many of the latest android devices are powered by chips like snapdragon gen2. We find that they need different optimization strategies.
 - We need to connect our LLM runtime to different host languages. In iOS we need to interface with object-c and swift. We will need to support java to enable the android ecosystem

<p align="center">
  <img src="/blog/img/android/android-vs-ios.png" width="50%">
</p>


Thanks to MLC-LLM’s universal deployment solution, we can overcome these challenges and productively deploy a vicuna 7b model onto a Samsung galaxy S23, powered by the latest snapdragon gen2.

<p align="center">
  <img src="/blog/img/android/android-diagram.png" width="80%">
</p>

The cornerstone of our solution is machine learning compilation (MLC), which we leverage to efficiently deploy AI models.
 - We effectively reused the same ML compilation pipeline for overall model ingestion, fusion and memory planning.
 - We leveraged TensorIR, and our automatic optimization stack of TVM unity to generate and optimize specific GPU kernels for Adreno GPU on the snapdragon chip via generating OpenCL kernels.
 - Thanks to the python first development flow, we can productively iterate on model weight encoding, decoding and kernel computation to get reasonably good performance out of the LLM.

Our solution provides a good harness to further optimize more models on android hardware backends. We believe there are still a lot of opportunities but it is amazing how far we can go in one week’s effort. We would love to work with the open source community to bring further optimizations via ML compilation.

Because android does not have the 4GB app RAM limit which is enforced by iOS, we can leverage more RAM than our iOS deployment. So we choose to enable a 4 bit quantized vicuna model. This gives the model more capabilities, especially in other languages than English. We are also looking forward to supporting other storage efficient models in the future.

You can check out [the demo instruction](https://mlc.ai/mlc-llm/#android) to try it out, our [github repo](https://github.com/mlc-ai/mlc-llm) for source code. MLC LLM enables deployment to various devices, including windows, linux, mac iPhone, and now android. You are also more than welcomed to checkout the project page for more information on running on other devices.


## Acknowledgement

The MLC LLM project is initiated by members from CMU catalyst, UW SAMPL, SJTU, OctoML and the MLC community.

We would love to continue developing and supporting the open-source ML community. The overall MLC projects are only possible thanks to the shoulders open-source ecosystems that we stand on. We want to thank the Apache TVM community and developers of the TVM Unity effort.  The open-source ML community members made these models publicly available. PyTorch and Hugging Face communities that make these models accessible. We would like to thank the teams behind Vicuna, SentencePiece, LLaMA, and Alpaca. We also would like to thank the OpenCL Vulkan, Android, C++, python Rust communities that enable this project.
