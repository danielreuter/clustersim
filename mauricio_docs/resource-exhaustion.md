This document contains a sketch of a protocol for ensuring that a compute cluster in a given time period is:
1. Executing some declared program correctly
2. Not executing some prohibited program
We aim to achieve (2) by ensuring that the cluster’s computational resources (compute, memory, bandwidth) are sufficiently exhausted by (1).
# Big picture goal: compute verification
For AI to go well, we will likely need to develop a system by which states can verify one another’s compliance with global rules governing the use of compute. Below are some types of international agreements this system could be tasked with verifying:
- **Training pause**: Compute cannot be used to conduct training, however defined.
- **Safe deployment requirements**: Compute cannot be used to serve frontier AI systems unless they have been trained in certain ways and/or have passed certain evals.
- **Prohibition on automated offensive operations**: Compute cannot be used to stage large-scale cyber attacks or interfere in political processes.
To facilitate this, we are imagining that users of compute (provers) will stream information about their workloads in some format to one or more verifier nodes operated by foreign nations, which then check that these claims have the following properties:
1. **Correctness**: Virtually all declared workloads were performed as claimed.
2. **Comprehensiveness**: Virtually no undeclared workloads were performed concurrently.
3. **Compliance**: Virtually all declared workloads were compliant with the given ruleset.
Ideally these properties would be verified in **zero-knowledge**—though alternatively they could be verified using mechanisms that tightly upper-bound the amount of private information that could be leaked.
# Problem: it is difficult to prove negative claims about compute use
In this paper, we will assume that solutions to verifying correctness and compliance exist, and instead will focus on **verifying comprehensiveness**—that is, that a given set of computing devices reported all relevant activities they performed in a given time period.
This problem is especially difficult in our setting because **states may not trust one another’s** [**TPMs**](https://en.wikipedia.org/wiki/Trusted_Platform_Module) **or** [**TEEs**](https://en.wikipedia.org/wiki/Trusted_execution_environment)**,** which means that we will not be able to rely on the technologies that we typically use today to ensure that accelerators are only doing what they should be doing. This is because:
- States could pressure chip vendors to share their attestation keys (which nullifies their security guarantees)
- It may be hard to verify that hardware is free of backdoors
- Even if states can develop new, mutually trusted accelerators, there will likely still be millions of previous-generation accelerators already deployed that still cannot be trusted
# Proposed solution: resource exhaustion protocols
Below we sketch out a solution that requires only that:
1. We send in **international inspectors** to each datacenter to determine the resource capacities of the devices it contains
2. We place a **single trusted pinging device inside of each datacenter** and use it to periodically challenge accelerators to quickly respond with measurements of the state they currently have in memory
This proposal is thus **qualitatively faster and cheaper to set up in the event of an international agreement** than [other proposals](https://www.longview.org/request-for-proposals-on-hardware-enabled-mechanisms-hems-for-ai-verification/), which—in addition to requiring (1), plus some workload reporting scheme—require making extensive physical modifications to all existing compute infrastructure. We are interested in seeing how far we can take this proposal because it seems like an attractive first resort. 
[diagram]
## Definition
An $(X, D)$**-resource exhaustion protocol (RXP)** is an interactive protocol between a developer (prover) and an untrusting third party (verifier) that aims to attain the following security property (or immediately abort):
> $(X, X^*, D)$**-exclusion**: The prover could not have executed some prohibited program $X$ concurrently with some declared program $X^*$ using devices $D$ in a given time period because $D$ had insufficient time or memory to do so
> 
To illustrate, we could set the following parameters:
- $X$ is 100 forward-then-backward passes using Llama-3-8B, arbitrary inputs
- $X^*$ is 1,000,000 forward passes using Llama-3-70B, arbitrary inputs
- $D$ is an $8 \times$ H100 node
The better an RXP ensures that $X^*$ exhausts $D$’s time and memory, the larger the set of prohibited programs it is secure against. 
## Why might this work?
1. Existing tools can verify that $X^*$ was correctly executed with some reasonable degree of confidence, e.g. subsampled re-execution or (heavily) subsampled zero-knowledge proofs
2. We may be able to **tightly upper-bound** the true memory capacity and network bandwidth of the devices $D$ via physical inspection and random disassembly, and **loosely upper-bound** the true arithmetic speed
3. We may be able to **tightly** **lower-bound** $X^*$’s true memory utilization over time using a **simple spot checking mechanism**
4. (2) and (3) suggest that the correct execution of $X^*$ might **leave insufficient free compute/memory for the full execution of $X$
## A simple setting
- The prover has one accelerator with memory capacity $M_\text{hbm}$
- The prover’s accelerator can run $N$ operations in the given time period
- The prover has a large cache with memory capacity $M_\text{cache}$
- The bandwidth between the cache and the accelerator is $B$
## Verifier affordances
- Knows $N, M_\text{hbm}$, $M_\text{cache}$, and $B$
- Knows exactly when the prover is *supposed* to execute each declared operation
- Does not know what algorithm the prover actually uses—it can only estimate some of its properties
- Has a low latency communication channel to the prover—though it can’t tell which device responds (unless it infers this from other information)
- Has a mechanism that allows it to check the correct execution of arbitrary programs (for now just assume this is perfect, and that computations are deterministic)
## How can we tell if devices are running a workload as claimed?
Generally, a prover might want to make two kinds of claims:
- A **logical statement**: e.g. $f(x)=y$
- A **physical statement**: e.g. this device ran $y \leftarrow f(x)$ using a particular physical process in a particular time period
A conventional way to prove a logical statement is to [produce a witness](https://en.wikipedia.org/wiki/Proof_of_knowledge) to that statement—some object that makes it clear that the logical statement is true. In our setting, provers will be asked to produce **physical witnesses**—objects that make it clear that some *physical statement* is true. 
The two main physical witnessing mechanisms [proposed in the literature](https://arxiv.org/abs/2507.15916) are:
1. **Network taps**: Trusted devices that attest the size/contents of messages passed between nodes (or even [between HBM and compute units](https://www.flexheg.com)). 
2. **Analog sensors**: Trusted devices attached to accelerators that measure e.g. the power draw or radio frequency. 
In this paper, we propose a third kind: **memory spot checks:**
1. A trusted pinging device is inserted into the datacenter
2. It is used to periodically challenge accelerators to quickly respond with measurements of the state they currently have in memory
3. These measurements are accumulated in some local buffer
4. They are checked offline as part of a broader system that checks the prover’s *logical claims* for correctness—e.g. the prover might be asked to re-execute a slice of its declared program and then issue a ZKP of correctness
There are **two challenge-response** implementations we are considering:
- **Verifier reads random state slices via RDMA [~1-5 microseconds]**: The verifier can read out random tensor slices directly from HBM. These should likely be of some ephemeral state like activations, which are more physically informative per bit than slices of models.
- **Prover sends a random projection of some of its state [~10-30 microseconds]**: The prover can respond with a compute-intensive measurement of some of its state—this could be useful if this is a computation that accelerators can perform much faster than CPUs, because it provides more evidence about *which device responded to the challenge*.
For reference, executing a DeepSeek-v3 MLP in decode on an H200 takes **~500 microseconds**. So in this setting the prover is meaningfully computationally bounded. 
Some notes:
- Technically we can only enforce that the adversary’s state at checkpoints is something like $(X^*, T)$**-pseudohonest**
    
    That is, we only enforce the condition that state consistent with $X^*$ at that checkpoint can be reconstructed within $T$ time steps. But this does not mean that state is correct:
    
    - Adversary could swap columns in matrices
    - Adversary could evaluate operations in unexpected orders
    
    **Open problem**: We currently don’t know how to lower bound the size of $(X^*, T)$-pseudohonest state—this seems crucial. 
    
- BOTEC on down-projection latency
    
    **Typical honest response latency for a down-projection response is ~10-30 microseconds**
    
    - GPU kernel launch and scheduling: **~40–50%**
    - In-HBM read of the challenged slice: **~30–40%**
    - Transmission of the down-projected fingerprint to the verifier: **~10–15%**
    - Miscellaneous RDMA/OS jitter: **~10–20%**
    
    [Numbers pulled from here.](https://www.notion.so/Prose-Compute-transparency-via-verifiable-memory-exhaustion-27f60f987af080219c08f36894cf4e31?pvs=21)
    
## Attacks/defenses
In general, the adversary can **trade off time for space**, and the better we can defend against this, the finer-grained the $X$ we can exclude. 
1. **Recomputation attacks**: The adversary might be able to quickly recompute honest state from very different starting states.
    1. Need to ensure that many $X^*$ **nodes are deep relative to** $T$, and that some or all of them have **transparent space/time requirements**
2. **Caching attacks**: The adversary can push intermediate states to the cache, freeing up HBM. Defenses:
    1. Fill up the **cache with synthetic state** and require proofs of space
    2. Measurement function can serve as **proof-of-parallel-compute** + ensure that cache ↔ accelerator bandwidth is sufficiently low to **prevent fast-fetching**
    3. Break caching with **time-dependent perturbations** to the declared workload—e.g. rotating the seed used for sampling from distributions, or injecting noise vectors 
3. **Compression attacks**: The adversary can compress its intermediate state to free up space. 
    1. Ensure the program is being **executed at a plausible rate**—this breaks the economics of compression/decompression by making amortization less effective.
    2. Ensure the intermediate states are **minimally compressible** by setting quantization requirements or adding noise in places. 
## Abstract protocol
An $(X, D)$**-RXP** consists of three phases:
**Declaration phase**: 
1. The prover declares some program $X^*$ that it would like to evaluate, along with some checkpoints (certain time steps at which it might get challenged, e.g. after every fused kernel)
2. The verifier checks that any algorithm that produces $(X^*, T)$-pseudohonest state at all checkpoints has a memory utilization curve that renders it infeasible to concurrently evaluate $X$ on $D$—otherwise, it rejects
**Execution phase**: Ensures that $D$ holds pseudohonest state at checkpoints by requesting measurements at a random subset of checkpoints and expecting a response within $T$ timesteps. 
**Verification phase**: Requests proofs of measurements and checks them. 
## Construction using GPT-2
[…]