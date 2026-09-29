---
layout: post
title: "TIRx Harness: An Open Compiler Harness for Agentic GPU Programming"
date: 2026-09-29 14:30:00 -0400
author: MLC Community
notitle: true
---

<style>
.container > h1 {
  font-size: clamp(1.75rem, 3vw, 2rem);
  margin-top: 1rem;
}

.content p:not(.post-meta) {
  margin-bottom: 1.75rem;
}
</style>

<p align="center">
    <img src="/img/tirx-harness/overview.png" alt="TIRx Harness architecture: the agent workflow connects to a knowledge base, compiler analyses, the TIRx foundation, and a benchmark server, with evolution traces and optimized kernels feeding self-improvement." width="1000" style="max-width: 100%; height: auto;">
</p>

**TL;DR:** We built TIRx Harness, a compiler harness combining a minimal stable compiler foundation, a knowledge base, tools, and a benchmark server to help agents develop correct, fast GPU kernels. On evaluated workloads, Kimi Delta Attention (KDA) kernels achieve geometric-mean speedups of 2.94× over FlashKDA (forward) and 6.84× over Flash Linear Attention (FLA) (backward).

Developing high-performance GPU kernels is an iterative engineering process: exploring implementations, drawing on hardware knowledge and existing code, and refining the result through debugging and measurement. Coding agents can already automate meaningful parts of this process, but in our experiments on workloads such as KDA, much of their effort went into work around the optimization itself: predicting how kernel code would be lowered to hardware behavior, diagnosing failures that could be timing-dependent or measurements distorted by concurrent GPU activity, and finding an implementation or optimization strategy relevant to the specific task. As a result, token efficiency drops sharply: the agent spends its budget resolving uncertainty around the optimization rather than exploring the optimization itself, so otherwise viable optimizations may never be reached within a practical search budget.

These were not simply search problems that could be solved by asking the agent to try more variants. They pointed to missing pieces in the development environment: a predictable programming foundation, tools for reasoning beyond a single execution, reusable implementation knowledge, and controlled evaluation.

We view these pieces together as a compiler harness: the agent-facing environment around a compiler foundation, including domain-specific analyses and diagnostics, an IR-specific knowledge corpus, and benchmarking and evaluation tooling. As agentic programming shifts more of the work from writing code to shaping the environment in which agents operate, we expect this layer to become increasingly important. This framing is inspired by several concurrent efforts in AI-oriented system design. Among them, [CAKE](https://arxiv.org/abs/2608.12629) makes compiler–agent co-design explicit, while [OpenAI’s Jalapeño](https://x.com/cdleary/article/2094878051238887834) reflects a related full-stack principle: designing the programming target itself to be clear and predictable enough for AI to optimize effectively.

TIRx Harness is one instance of this idea for GPU kernel development. For us, that starts with a central question: how can we design a compiler foundation that is minimal, stable, and extensible for agentic programming? Around this foundation, TIRx Harness combines a knowledge base, correctness and performance tools, and a benchmark server, giving agents a more reliable environment for kernel optimization. Across the evaluated KDA, MiniMax Sparse Attention (MSA), Multi-head Latent Attention (MLA), and Video Sparse Attention (VSA) families, reported family-level geometric-mean speedups range from 1.33× to 6.84×; KDA forward and backward reach 2.94× over FlashKDA and 6.84× over FLA, respectively.

## Building TIRx Harness around reliable iteration

These components address different sources of uncertainty in the same optimization loop. TIRx Harness provides the compiler-facing environment, while agent workflows use these capabilities to organize their search. We designed each component around a recurring failure mode we observed during agent-driven kernel optimization.

### TIRx Foundation: Make the Compiler Predictable

The programming foundation determines how directly an agent can connect what it writes to what the GPU executes. High-level abstractions are valuable for productivity, but in our experiments they could make optimization harder to reason about. An agent might know the execution strategy it wants to try without knowing what source program will make the compiler lower to that strategy. A long compilation path adds another source of uncertainty during debugging: when a numerical failure appears, the agent must consider both the kernel it wrote and the compiler transformations applied to it.

We address these sources of uncertainty in two ways. First, we keep TIRx close to the hardware. Its thin PTX-level programming surface lets agents express instructions and execution strategies more directly, rather than relying on a high-level abstraction to lower into the behavior they want.

Second, TIRx is intentionally minimal. It introduces as little independent semantics as possible beyond the hardware interface, keeping the IR and compilation path small and stable. With such a thin abstraction layer, the PTX ISA serves as the primary source of truth for instruction semantics. This also makes the system easier to extend: exposing a new PTX capability usually requires only a lightweight addition to the programming surface, rather than a new high-level abstraction or a larger compiler stack. We still retain basic structure such as loops and conditionals so the rest of the harness can analyze the program.

### Tools: Provide detailed feedback beyond benchmarking

Even with a simpler programming foundation, failures can still be hard to diagnose. A kernel may pass many runs and then fail because of an intermittent synchronization bug or data race, leaving the agent with little more than a pass/fail signal. This motivated a broader principle for the harness: tools should provide useful feedback about program behavior, not just test outcomes.

We built synchronization and data-race analyses to inspect concurrency behavior directly. When GPU execution is unavailable, numerical simulation provides another source of feedback by simulating numerical behavior without a GPU. For performance diagnosis, we integrate existing profiling tools including NCU and IKET. Together, these tools give agents more information to decide what to investigate and change next.

### Knowledge base: reuse known ideas, discover new ones

A predictable programming foundation and better diagnostics still do not give agents the implementation and optimization knowledge needed for a new kernel. In early runs, agents spent substantial effort learning the programming model and rediscovering useful optimization patterns.

To avoid relearning these ideas in every run, a natural approach is to summarize lessons from previous runs. In our experiments, however, agent-written summaries often overfit the case that had just been debugged and lose useful implementation context. We therefore preserve concrete implementations instead.

The kernel zoo is the core of this knowledge base. With more than 60 TIRx kernels, it gives agents concrete starting points and reusable optimization patterns. But the zoo only captures ideas that have already appeared in existing kernels. PTX documentation complements it by exposing hardware capabilities that can open new optimization directions. The zoo helps reuse known ideas; the ISA helps discover new ones.

### Benchmark server: make measurements comparable

Correctness tests, timings, and profiles decide whether a candidate is accepted and which optimization the agent tries next, so a misleading measurement affects many attempts rather than one. Two goals for the harness made such measurements harder to keep reliable when the agent launches the GPU work directly from its own environment. The first is scale: we want many agents running concurrently, but once they share a GPU, a timing change may reflect another agent's activity rather than the candidate's own improvement or regression, and the agent can no longer attribute a performance change to its edit. The second is platform coverage: we want to target hardware beyond datacenter GPUs, including edge devices such as Thor, but these platforms are not convenient hosts for a coding agent, so tying the agent to the machine it optimizes for would limit the hardware it can reach.

Both point to the same design: separate evaluation from the agent. TIRx Harness therefore uses a remote evaluation architecture built around KCoral, our benchmark server (blog coming soon). KCoral owns the GPUs, and whenever the agent needs one, whether to test correctness, benchmark a kernel, or collect a profile, it sends a request to KCoral instead of launching work locally. KCoral schedules all requests centrally so that the execution of one never affects another, and returns the measurements and diagnostic artifacts when the request completes. The agent gets a controlled, trusted environment for every measurement, and the harness gains new targets by attaching them to KCoral rather than by making every agent host run on them.

## What TIRx Harness enables

The components above matter because they change what an agent can do during optimization. Beyond generating more candidates, the harness can help agents discover hardware capabilities they had not considered, transfer strategies across kernels, and diagnose unsafe candidates with more than pass/fail feedback. The traces below illustrate these roles in practice; the next section evaluates the resulting kernel performance.

### Discovering new directions

Hardware documentation can expand the agent's search space, not just answer syntax questions. In one MSA sparse-prefill trace, the agent found tcgen05.mma output-lane masking in the PTX ISA and turned it into a useful optimization. The mask did not reduce the MMA work; instead, it increased the observed GPU frequency, yielding a 2–3% gain.

TIRx's hardware-close programming surface also lets agents act directly on hardware-resource tradeoffs. In KDA backward, the agent moved transposes, diagonal scaling, and a row-wise dot product onto the Tensor Core path. For the dot product, it computed a full 64×64 matrix and kept only the diagonal, deliberately spending extra Tensor Core work to remove shared-memory partials and reduction work from busy compute warps. This counterintuitive tradeoff—doing more arithmetic on an underused hardware path to relieve the bottlenecked one—improved latency by about 15%.

### Transferring strategies

The kernel zoo gives agents concrete strategies that can be adapted rather than copied literally. In MSA, the agent adapted FlashAttention-4's mixed native/software exponential strategy, using ordinary FMA work to relieve the native exp path and improving latency by about 2.9%.

In KDA forward, the agent transferred Gated DeltaNet (GDN)'s hierarchical inverse decomposition, replacing a serial 32×32 recurrence with small block inverses and MMA-based merges and reducing latency by about 17%. These cases show why we preserve concrete implementations: they expose decompositions and dataflows that can transfer across different kernels.

### Diagnosing unsafe candidates

Analysis tools can catch correctness issues that ordinary testing may miss. In one KDA trace, a candidate passed the benchmark’s correctness tests, but synchronization analysis still identified a latent bug: a reused mbarrier could advance to its next generation before a late consumer finished waiting on the previous one. The agent could then repair the synchronization protocol before this timing-dependent issue surfaced in execution.

Together, these traces show three ways the harness changes the optimization loop: the ISA expands the search space, the kernel zoo provides transferable strategies, and analysis tools turn unsafe candidates into concrete diagnoses. The benchmark server then provides controlled performance feedback for deciding what to keep.

## What agents achieved with the harness

The evaluation asks whether the capabilities above translate into fast kernels. We evaluate KDA, MSA, MLA, and VSA. All experiments reported here were run on NVIDIA Blackwell GPUs using [Humanize 2’s flame chase workflow](https://humanfia.ai/), with web access disabled during optimization. The baseline implementations use the versions recorded in the September 25–26, 2026 curated sweep. Each family is compared against its own optimized reference implementation using GPU kernel time rather than end-to-end application latency.

<p align="center">
    <img src="/img/tirx-harness/benchmark-results.png" alt="Agent-evolved kernel results on NVIDIA Blackwell GPUs: geometric-mean speedups of 2.94× for KDA forward, 6.84× for KDA backward, 2.59× for MSA prefill, 3.99× for MSA decode, 1.33× for KDA decode, 1.71× for MLA, and 1.68× for VSA, with min–max ranges." width="1000" style="max-width: 100%; height: auto;">
</p>

Across these workloads, agents using TIRx Harness produced kernels that are competitive with—and often faster than—the reference implementations. Reported family-level geometric-mean speedups range from 1.33× to 6.84×. The figure shows the min–max range across evaluated configurations together with each family’s geometric mean.

The agentic programming landscape is evolving quickly, and benchmark results can change on the scale of days as agents, workflows, compilers, and reference implementations improve. Our goal is therefore not to establish a permanent ranking of approaches, but to demonstrate what TIRx Harness can enable. Because the reference differs across families, the speedups should be interpreted within each family rather than as a cross-family ranking. The kernels are available [here](https://github.com/mlc-ai/tirx-kernels).

## Closing the loop

Kernel evolution can improve the harness as well as the kernel. When a run exposes a tool bug, unsupported behavior, or a missing check, its trace provides a concrete reproducer that can be used to improve the analysis tools. Successful kernels follow a different path: they enter the kernel zoo as better starting points and reusable optimization strategies for future runs.

This is how we think about self-improvement in TIRx Harness: useful outcomes from one evolution run become part of the environment available to the next. Rather than asking the agent to summarize what it learned, we preserve concrete artifacts—a validated tool improvement or a measured kernel implementation—that later runs can directly build on.

## What's next

We see three directions for the next stage of TIRx Harness. First, we want to evaluate how quickly the harness can adapt to new GPU architectures, including how rapidly agents can make use of new instructions and hardware features. Second, we want to push TIRx further toward the hardware, exposing lower-level mechanisms that give agents more direct access to performance opportunities. Third, we want to extend our correctness tooling to megakernels, where validating synchronization, memory accesses, and numerical behavior becomes harder as more computation is fused into a single program.

## Acknowledgments

We thank the NVIDIA CAKE team, Kernel Design Agent (KDA) team and SOL-ExecBench team for helpful discussions and feedback throughout this work.

## Getting started

To try TIRx Harness, start with the [documentation](https://tirxharness.mlc.ai/docs/), which covers installation, running kernel optimization tasks, and using the harness's analysis and remote execution tools. Our book, [Agentic GPU Programming for MLSys](https://mlc.ai/agentic-gpu-programming-for-mlsys/) introduces the main elements of agentic gpu programming and compiler harness.
