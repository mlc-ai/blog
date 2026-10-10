---
layout: post
title: "KCoral: Lightweight Benchmark Server for Agentic GPU Programming"
date: 2026-10-05 09:00:00 -0400
author: MLC Community
notitle: true
---

Machine learning systems are becoming increasingly important with the rapid development of AI, and GPU kernels are central to these systems. AI agents can automate GPU kernel development by repeatedly generating code, running it, checking correctness and performance, and refining it. Recent efforts such as [CAKE](https://arxiv.org/abs/2608.12629) and [Kernel Design Agents (KDA)](https://github.com/NVlabs/kda) have demonstrated the growing capability of agents for GPU kernel development.

With the rapid growth of kernel agent workloads, a new challenge arises: efficiently managing and sharing GPU resources. Agents need GPUs only during the evaluation part of the loop, so assigning a GPU to every agent can waste capacity. Sharing GPUs improves utilization but requires coordination to avoid interference between evaluations. As workloads scale, GPU resources may grow from a single GPU to multiple GPUs and nodes, requiring efficient management and scheduling. This is especially important in **kernel agent experiments** and **reinforcement learning for kernel agents**, where many agents or rollouts may run concurrently.

To address this challenge, we designed **[KCoral](https://kcoral.mlc.ai/)**, a lightweight benchmark server that decouples GPU execution from the agent loop. Agents generate and revise code in their own workspaces, then send evaluation requests to KCoral. The server schedules GPU access, executes the requests, and returns results for the next iteration. A standardized protocol supports diverse kernel workloads, and agents need no local GPU. KCoral provides:

- **Shared GPU access:** many agents can share a small pool of GPUs.
- **Seamless scaling:** a unified entry point supports single-GPU, multi-GPU, and multi-node execution.
- **Reliable and efficient evaluation:** request-level isolation limits interference, with performance close to local evaluation.
- **Edge development:** agents can use GPUs on remote edge devices such as NVIDIA Jetson Thor.

KCoral provides a shared foundation for agentic kernel development and GPU agent post-training, enabling agents to improve machine learning systems at scale.

<p align="center">
    <img alt="KCoral connects the agent's kernel development loop to a remote protocol and an engine that schedules isolated GPU evaluations." src="/img/kcoral/overview.png" style="max-width: 100%; height: auto;" width="900"/>
</p>

## A Standardized Protocol for Remote Kernel Development

In the local agentic kernel development loop, an agent generates a kernel, evaluates it locally, and revises it. This binds the agent to a GPU, which makes the GPU hard to share and does not work when the GPU is remote.

We designed the KCoral protocol to support remote kernel development. A local client sends a request that follows the protocol to a server that owns the GPUs, and the server returns the results. The protocol clearly describes the kernel, evaluation, or debugging workload, so it provides a unified interface across different devices and GPUs.

The protocol is designed as an atomic program over HTTP. It contains a sequence of operations that upload code and data, run code, or return results. The server returns the specified results or an execution error, together with the logs. The protocol has four basic operations:

| **Operation** | **Description** |
| --- | --- |
| `upload` | Upload source code, a tensor, bytes, a compiled library, or a file. |
| `get_function` | Select a named function or object from an uploaded module or library. |
| `run` | Call a function with arguments and bind the value it returns. |
| `return` | Select an earlier value, file, or folder for the response. |
{: .table }

For example, the following program uploads a Python module and a tensor, runs `add_one` on the server's GPU, and returns the result:

```python
import numpy as np
from kcoral import Client, Program

SOURCE = """
def add_one(x):
    return x + 1
"""

program = Program()
module = program.upload(kind="module", source=SOURCE)
add_one = program.get_function(module=module, name="add_one")
x = program.upload(kind="tensor", value=np.arange(4, dtype=np.float32))
y = program.run(fn=add_one, args=[x])
program.return_(key="output", value=y)

with Client("http://localhost:8000") as client:
    result = client.execute(program)
print(result.results["output"])  # [1. 2. 3. 4.]
```

With this protocol, remote development becomes easy: for example, launch a server on an NVIDIA Thor and develop kernels on your MacBook. See the [example](https://github.com/mlc-ai/kcoral/tree/main/examples/thor) for more details.

We welcome the community to adopt the protocol and build tools and servers on top of it.

## KCoral Engine: Scaling Kernel Evaluation Across GPUs and Nodes

<p align="center">
    <img alt="The KCoral router connects multiple GPU nodes; each server caches uploads and schedules isolated workers with exclusive GPU access." src="/img/kcoral/engine.png" style="max-width: 100%; height: auto;" width="900"/>
</p>

The KCoral engine implements the protocol. It manages the GPUs on a machine and schedules each request to the least busy GPU. Once scheduled, the program has exclusive GPU access. This ensures that multiple agents can use the same GPU to evaluate kernels with minimal interference.

To support multiple GPU nodes, we further designed a Rust-based KCoral router that provides a unified entry point and assigns requests according to node health and load. It also manages the lifecycle of the servers and restarts them when they crash. Therefore, even with many concurrent agent requests, KCoral can easily distribute them across the large number of GPU nodes it manages.

To prevent an agent from obtaining information about other unrelated agents, or from modifying system files to achieve reward hacking, KCoral has an isolation mechanism: each request can only access a fixed set of system directories and its own working directory. To minimize overhead, KCoral uses bubblewrap, which achieves millisecond-level startup overhead and avoids the second-level overhead of traditional Docker-based isolation.

When evaluating many different implementations of the same kernel, the scripts, tensors, and other dependencies of each evaluation request are usually identical, and only the kernel differs. Re-uploading all files every time is wasteful. We therefore implemented a caching mechanism: when a file with the same hash is uploaded again, the duplicate upload is skipped automatically.

The KCoral engine supports both single-GPU and multi-GPU kernels. When submitting a multi-GPU kernel, you only need to specify the number of GPUs required, and the scheduler automatically waits for and allocates enough GPUs.

## CPU-GPU Decoupling: Freeing GPUs During Compilation

KCoral can separate CPU compilation from GPU execution.

Kernel evaluation does not need the GPU throughout. In particular, the compilation step could be very slow and does not need a GPU at all. If a request holds the GPU for its entire duration, the GPU sits idle in the middle and is wasted. KCoral provides two mechanisms to prevent compilation from occupying GPU resources: releasing the GPU in the same program, or launching two programs, one for compilation and another for execution.

<p align="center">
    <img alt="Releasing GPU access during CPU compilation or using a separate CPU server overlaps compilation with GPU execution." src="/img/kcoral/cpu-gpu-decoupling.png" style="max-width: 100%; height: auto;" width="900"/>
</p>

We provide a `cpu_only` flag for `get_function` to mark the parts of the program that do not need a GPU, so the GPU is released while they run. The GPU can then be used by other programs that need it:

```python
program = Program()
ops = program.upload(kind="module", source=OPERATIONS)

# Compilation runs without the GPU; the GPU is free for other programs.
compile_kernel = program.get_function(module=ops, name="compile_kernel", cpu_only=True)
library = program.run(fn=compile_kernel, args=[KERNEL_SOURCE, {"arch": "sm_100a"}])

# Loading and benchmarking reacquire the GPU.
load_kernel = program.get_function(module=ops, name="load_kernel")
benchmark = program.get_function(module=ops, name="benchmark")
kernel = program.run(fn=load_kernel, args=[library])
timing = program.run(fn=benchmark, args=[kernel])
program.return_(key="timing", value=timing)
```

In large-scale agent or post-training experiments, the imbalance between CPU compilation and GPU execution becomes more severe. A better approach is to separate them into dedicated CPU and GPU services. KCoral supports remote compilation by letting CPU-only servers compile kernels and return the compiled artifacts, which are then sent to GPU servers for execution and benchmarking. This allows the agent, compilation, and GPU execution clusters to scale independently and adjust their resource ratios for different workloads.

```bash
# On the CPU host
kcoral server --device cpu --num-workers 8 --host 0.0.0.0 --port 8000
# On the GPU host
kcoral server --device gpu --gpus 0 --host 0.0.0.0 --port 8001
```

```python
with Client(CPU_URL) as cpu_client, Client(GPU_URL) as gpu_client:
    arch = gpu_client.target()["arch"]
    # Request 1: compile on the CPU server and return the shared library.
    compiled = cpu_client.execute(compile_program(arch))
    # Request 2: upload the library to the GPU server, check, and benchmark.
    result = gpu_client.execute(benchmark_program(compiled.results["library"]))
print(result.results["check"], result.results["timing"])
```

Notably, this can make kernel evaluation on KCoral faster than simple local evaluation because it can overlap and parallelize compilation with the execution of other kernel evaluation tasks. Read the [remote compilation tutorial](https://kcoral.mlc.ai/docs/latest/tutorials/remote-compilation.html) for more details.

## KCoral CLI

To simplify the use of KCoral, we designed the KCoral command-line interface (CLI), which runs common kernel development command-line tools on a KCoral server with the same experience as the local tools. The CLI currently wraps `python`, `compute-sanitizer`, `ncu`, `run-iket`, and `shell`, so a developer or an agent can run a correctness check, hunt for memory errors, or collect an Nsight Compute report on a remote GPU without changing how they invoke the tool. For example:

```bash
# Point every kcoral command at the GPU server.
export KCORAL_URL='http://gpu.example.com:8000'
# Upload the experiment/ directory and run check.py with the worker's Python.
kcoral run python --send experiment -- experiment/check.py
# Run the same script under compute-sanitizer to catch CUDA memory errors.
kcoral run compute-sanitizer --send experiment -- python experiment/check.py
# Profile capture.py with Nsight Compute and download the report to artifacts/ncu.
kcoral run ncu --send experiment --out artifacts/ncu -- --set basic -- python experiment/capture.py
# Run an arbitrary shell script in the remote working directory.
kcoral run shell --send experiment -- bash experiment/setup.sh
```

Every command has the form `kcoral run TOOL [KCoral options] -- [native arguments]`: the options before `--` select the server and the files to send, and everything after it is passed unchanged to the tool on the worker. `--send` uploads a script or a whole directory into a fresh remote working directory, and `--fetch` with `--out` downloads the files it produces, such as profiler reports.

See [Builtin CLI Tools](https://kcoral.mlc.ai/docs/latest/client-guide/builtin-cli-tools.html) for the options and the guide to each tool.

## Evaluation

We evaluate KCoral along three dimensions: **fidelity** of kernel timing, **throughput** as more agents share a GPU, and **robustness** when candidate programs fail. All evaluations use a single NVIDIA B200.

**Fidelity: preserving performance feedback.** Kernel optimization agents need reliable timings to guide optimization. Across 500 workload configurations from 82 kernel families in [TIRx-kernels](https://github.com/mlc-ai/TIRx-kernels), we compare KCoral with direct local timing using the same GPU, inputs, and timing settings, including L2 cache flushing. We report `100 × |remote − local| / local`. Measurements cover kernel execution only, excluding compilation, input preparation, and request latency.

| **Difference from local timing** | **Result** |
| --- | --- |
| Median | 0.425% |
| 95th percentile | 3.136% |
| Workloads within 5% | 497 / 500 |
{: .table }

All three workloads above 5% are short root mean square normalization (RMSNorm) or layer normalization (LayerNorm) kernels, taking 1.9–2.2 μs locally, with absolute differences of only 0.13–0.38 μs. For example, one RMSNorm workload takes 2.079 μs locally and 2.272 μs through KCoral: a 0.193 μs difference becomes 9.28%. At this timescale, sub-microsecond differences can produce large percentage differences. Overall, KCoral closely matches local timings and preserves the performance feedback needed for optimization.

**Throughput: sharing GPU capacity.** We run 256 workloads: eight fixed TIRx-kernel specializations, including [Kimi Delta Attention](https://github.com/mlc-ai/TIRx-kernels/blob/8b6ed13/tirx_kernels/kda/kda_forward_portfolio_multishape.py), [Alpha-MoE](https://github.com/mlc-ai/TIRx-kernels/blob/8b6ed13/tirx_kernels/moe/alphamoe_fp8_blockscale_qwen3next.py), and [MiniMax Sparse Attention](https://github.com/mlc-ai/TIRx-kernels/blob/8b6ed13/tirx_kernels/msa/msa_prefill_multishape.py), repeated 32 times each. Each workload compiles from a cold cache, constructs GPU inputs, and benchmarks an existing kernel; agent generation is excluded. The local baseline runs the same workloads sequentially without KCoral. We vary request concurrency with an equal number of KCoral workers. At concurrency 32, up to 32 requests remain in flight, with a new request submitted as one finishes. Requests take turns using the GPU, while CPU preparation and compilation marked `cpu_only=True` can run concurrently with another request’s GPU work.

<p align="center">
    <img alt="On one NVIDIA B200, mixed-workload throughput rises from 4.98 to 12.86 workloads per minute, while matrix multiplication evaluation reaches 4.86 times local sequential throughput." src="/img/kcoral/throughput.png" style="max-width: 100%; height: auto;" width="900"/>
</p>

At concurrency 32, KCoral reaches 12.86 workloads/min, versus 4.98 locally, a 2.58× speedup. Operations requiring exclusive GPU access take 1,190.79 seconds in total, compared with a 3,083.61-second local batch time, giving a practical speedup bound of 2.59×. KCoral reaches 99.7% of this bound, leaving little room for further gains without reducing work under exclusive GPU access.

We also evaluate a shorter workload that compiles a C++ wrapper around a fixed 4096×4096 cuBLAS matrix multiplication, constructs inputs, checks correctness, and benchmarks the operation. With 16 persistent workers and 16 concurrent requests, KCoral achieves 4.86× the throughput of matching local sequential execution, measured across six batches of 96 repetitions.

Both evaluations exclude agent generation time. To model the full workflow, we add a 162-second generation delay before each evaluation, matching the approximate median of our Kimi Delta Attention forward trace from [TIRx Harness](https://blog.mlc.ai/2026/09/29/tirx-harness-an-open-compiler-harness-for-agentic-gpu-programming). We run 32 simulated agents with 16 KCoral workers, each submitting two evaluations. All 64 evaluations finish in 6.54 minutes, compared with an estimated 180.82 minutes for a naive sequential-agent baseline, a 27.63× workflow speedup. This gain includes both parallel generation and overlapping evaluation. KCoral can also distribute independent requests across multiple GPUs, allowing throughput to scale with available GPU and CPU capacity.

**Robustness: continuing after failures**. We collect 200 CUDA, Triton, and CuTeDSL kernel candidates covering matrix multiplication, attention, normalization, and other operators, including candidates from [PTXBench](https://arxiv.org/pdf/2608.17379) evaluation trajectories. The suite includes compilation and architecture rejections, numerical mismatches, runtime errors, and worker exits. KCoral returns the corresponding compiler, runtime, or correctness-checker diagnostics, and all 200 cases match their expected outcomes. After each case, we submit a known-good GPU workload; all 200 follow-up requests succeed on the same server instance, showing that agents can revise and resubmit failed candidates without restarting the server.

## Start Using KCoral

Install KCoral from pip (check out the [installation guide](https://kcoral.mlc.ai/docs/latest/getting-started/installation.html) for more methods):

```bash
pip install kcoral
```

Start a KCoral server on a device with a GPU. You can customize the host and port address. Please note that KCoral allows clients to execute arbitrary code on its workers, so make sure to only allow trusted clients to access your KCoral server or Router. Deploy on a trusted, isolated network and never expose these endpoints to the public internet. Run workers in a sandbox with restricted permissions and access to host resources.

```bash
kcoral --gpus 0 --host 0.0.0.0 --port 8000
```

You can start writing programs with the KCoral client and send them to the address you picked. The client does not need a GPU or a CUDA toolchain.

When connecting an agent, you can give it a specific task like this:

```text
Read .agents/skills/kcoral-client/SKILL.md.
Use the KCoral service at http://localhost:8000
to implement Triton addition for two float32 vectors of length 4096.
First check correctness against PyTorch, then measure performance after it passes.
Save a client program that can be rerun, and report the check results and execution time.
```

The agent still edits code in its original workspace, sends programs to the KCoral server when execution is needed, and continues making changes based on returned errors or performance results.

For the complete guide to using KCoral, please check out:

- [Website](https://kcoral.mlc.ai/)
- [GitHub](https://github.com/mlc-ai/kcoral)
- [Quick Start](https://kcoral.mlc.ai/docs/latest/getting-started/quickstart.html)
- [A tutorial on kernel benchmarking](https://kcoral.mlc.ai/docs/latest/tutorials/benchmark-kernel.html)

## Integrations

KCoral is built to integrate with the agent kernel harness and kernel agent RL environments.

[TIRx-harness](https://blog.mlc.ai/2026/09/29/tirx-harness-an-open-compiler-harness-for-agentic-gpu-programming) is a kernel agent compiler and harness. TIRx-harness uses KCoral as its default backend for kernel evaluation, supporting its large-scale kernel evolution experiments.

[Kernel Design Agents (KDA)](https://github.com/NVlabs/kda) will also integrate KCoral to enable remote compilation and GPU sharing across many agents.

## Acknowledgement

We thank the NVIDIA CAKE team, Kernel Design Agent (KDA) team, SOL-ExecBench team, and PTXBench team for helpful discussions and feedback on this post.
