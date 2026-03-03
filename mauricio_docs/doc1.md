Tentative take: Covert output should be kept to Kbps, and memory wipes don't help much
Mauricio Baker
February 25, 2026
Summary
(Feedback welcome! It’s very possible I’ve overlooked some factor that would change the bottom line by orders of magnitude, as happened during drafting. This doc is based on ~2-3 days of looking into this and writing it up.)

How tightly must verification bound secret interconnect bandwidth, to make secret AI training impractically slow (say, making it take 5 years)? Based on probabilistic back-of-the-envelope estimates, I tentatively argue:
To be confident that a frontier training run on known hardware will take 5 years, covert output bandwidth must be on the Kbps scale (maybe around 10 Kbps or 100 Kbps, ideally much lower for a safe buffer).
This will require nearly complete front-channel replicability (or small pods with tiny digital bandwidth) and, for most pods, elimination of all large side channels.
Input caps and memory wipes add little value to output caps, barring huge costs (e.g. 10x performance hits). Memory wipes and input caps seem likely to start seriously slowing training only at the point that output caps are already slowing it more severely—except maybe if we can execute full memory resets in seconds or accept > 50% downtime.

These results seem relatively pessimistic because they consider that, e.g.: gradients may be highly compressible, input bandwidths and inference pods may be big, and memory wipes may have to be relatively infrequent (around one per 10 minutes) to avoid >> 10% downtime.
Background
This assumes rough familiarity with the idea of network taps being used for inference verification (e.g. Memo – feasibility of network taps for treaty verification; Rinberg, et al., 2025), as well as “memory wipes,” plus concerns about covert side channels (How likely is it for a verifier to be sure an inference cluster is effectively airgapped [Link shared]).
Security arguments overview
Let’s consider the use of network taps for making it extremely inefficient to do frontier AI training on monitored hardware, which, if enough hardware is monitored, could essentially pause AI training. This goal seems like it should be “easy mode” for network taps, since covert frontier training requires lots of covert output bandwidth, at least more so than covert inference.

I’ll assume the goal is to force a frontier training run to take 5 years, since 5 years into an AI treaty seems long enough that probably either:
the training run will no longer pose a major threat, due to AI safety/resilience having had enough time to catch up to it, or
the treaty will have collapsed anyway due to unmanageable threats from dark compute.
(The results are similar if we instead assume, say, 2.5 or 10 years.)

There are a few security arguments we could make for why network taps would slow training:
Covert output bandwidth cap: Covert frontier training needs some total amount of covert output bandwidth (that is, a total out of each pod or data-parallel unit, whether via front or side channels, to sync gradients); rate-limiting these outputs rate-limits covert training. Note this simple argument makes no use of memory wipes.
Covert input bandwidth cap: Like the above, but for input bandwidth.
Caps + memory wipes: Recurring “memory wipes” slow pods’ accumulation of covert state (e.g., loading the latest model checkpoint, computing gradients). Thus, more covert state has to be redundantly re-computed and/or transmitted through capped covert channels, slowing down training.

Back-of-the-envelope calculations suggest that, roughly, only the covert output bandwidth cap matters. The other two arguments likely give much weaker bounds, unless maybe if we can achieve any of: low total input bandwidth, memory wipes that only take seconds, or accepting 50-90% downtime for frequent slow memory wipes.

(Throughout this piece, for consistency, I only use “bits” units, never “bytes.” So e.g. “Tb” is not a typo for “TB.”)
Why these pessimistic results?
This piece’s results seem more pessimistic than previous estimates because of incorporating more unfortunate factors or estimates (and not finding new factors that compensate):
Gradients may be very compressible, due to being sparse. This greatly reduces covert output bandwidth requirements.
Scher and Thiergart (2025, p. 127) also account for this, but this piece’s estimates are a little more conservative (also with regard to distributed training and floating point precision), seemingly accounting for the difference between our ~10-100 Kbps estimates and their preliminary recommendation of a 1 Mbps cap.
Capping covert input bandwidth seems hard, since input bandwidth is usually (much) bigger than output bandwidth, and inputs can’t be recomputed to check their legitimacy.
Inference pods may be big. As inference pods grow, the legitimate inference bandwidth grows, but the covert bandwidth needed to sync gradients stays the same, so it becomes easier to sneak training communication through legitimate bandwidth.
Scher and Thiergart (2025, p. 124) consider an inference pod with 128 H100s, which might be politically feasible but isn’t directly retrofittable onto today’s larger inference pods.
Each memory wipe probably entails downtime on the order of a minute, to ensure memory has been wiped and reload legitimate memory. So if we want to keep downtime around 10% or lower, each duration between memory wipes must be around 10 minutes—plenty of time to accumulate and use covert state.
For reference, one frontier training step typically just takes seconds, assuming a few months of training and a few million training steps.
Even if an inference batch or training step will be interrupted by a memory wipe, a prover can covertly output some of the gradients or activations to “save its work.” If the covert state can then be re-loaded quickly—as may be easy due to the above-mentioned difficulty of capping covert input bandwidth—then the prover might not have lost much from the interruption.
If outputting training activations, the amount that each pod needs to output shrinks as more pods are involved in data parallelism. (And RL may allow for other optimizations, like only outputting especially successful rollouts.)

In this piece, I also try to only consider precise, confident security arguments, not ones that are plausible but more speculative, like assuming that having to run some legitimate-passing inference would slow down training. Still, such delays might also succeed and might stack with the delays from these arguments.
Covert output bandwidth cap
Here is a simple estimate of the needed covert output cap, using point estimates:

gradient_data_size = 1e12 * 4  // 1T params, FP4
n_syncing_steps = 1e4  // a million training steps (estimate from LLMs), but only 1% 
need to be synced across pods (assuming fairly successful distributed training)
covert_output_compressibility_factor = 300  // estimate from LLMs (due to gradients 
being sparse and quantizable)
total_covert_output_needed_bits = gradient_data_size * n_syncing_steps / 
covert_output_compressibility_factor
covert_output_rate_needed_bps = total_covert_output_needed_bits / 1.6e8  // 1.6e8 
seconds is 5 years

The result is that the needed covert output rate is roughly 1 Mbps. Note we assumed little about the structure of a training workload: only that it requires syncing (potentially compressed) gradients some number of times, with these gradients needing to pass through the rate-limited output channels.

But that’s just what’s needed under the above roughly median (or a little conservative) assumptions; high confidence requires being more robust to possible parameter estimates. Incorporating uncertainty into the parameter estimates, we get a need for around 10 Kbps for 90% confidence, or around 100 Kbps for 70% confidence.

Even though we’ve incorporated parameter uncertainty, these results are still fairly brittle. Shifting a parameter distribution by an order of magnitude (which isn’t that hard to do given the amount of guesswork here) tends to change the results by an order of magnitude. So the appropriately uncertain takeaway seems to be: we should probably aim to limit covert output bandwidth to roughly the Kbps scale or ideally lower, rather than Mbps.
Covert input bandwidth cap V2 [in progress]
The covert input needed should be about the same as the covert output needed (since each model sync requires inputting the model weights and outputting the gradient, which are the same size pre-compression). However, the model weights should be less compressible than the gradients, since the former are less sparse. Per LLMs, a 10x compressibility factor seems conservative. Plugging this into the covert output bandwidth cap estimate without other changes, we get a needed covert input bandwidth cap of around 1 Mbps for 90% confidence, or around 10 Mbps for 70% confidence.

[The above and below aren’t yet synced.]

Unfortunately, with today’s largest inference pods, the covert input bandwidth seems likely to exceed the needed cap. A 1T parameter model with 10% active parameters needs 2*1e12*0.1 = 2e11 FLOP per token processed. Conservatively approximating utilization as 100% (as the workload may be prefill) and conservatively considering a large pod (Nvidia DGX SuperPOD with 8 DGX GB200 systems, 5.8e18 FLOP/s at FP4 without sparsity), a pod can process 5.8e18/2e11 ≈ 3e7 tokens/s. Supposing these are very efficiently transmitted at log2(vocabulary_size = 256,000) bits each and compressed by 1-10x, this is ~5-50 Mbps. Assuming the covert input bandwidth ≈ total input bandwidth (because it’s hard to detect malicious inputs—unlike with outputs, we can’t expect randomized recomputation to reproduce them), then the covert input bandwidth exceeds the needed cap.

However, the needed covert input bandwidth cap may be feasible for relatively small inference pods, namely ones ~10-1,000x smaller in throughput than Nvidia DGX SuperPODs (which can do 5.8e18 FLOP/s), by the above calculations. Such a cap would also require keeping covert side channels at << 1 Mbps.

(And even if training phased a large delay due to the covert input bandwidth cap, this delay wouldn’t add up with the delay from the covert output bandwidth cap.)
Covert input bandwidth cap [previous version]
This argument is similar to the above (covert output bandwidth cap), but two factors differ. Overall, a cap on covert input bandwidth seems much less useful than one for outputs, since the input cap that’s practical may be ~100-100,000x higher than what’s practical for outputs, while only being able to afford being ~100x higher. However, these aren’t carefully reached estimates; they may especially be wrong if it’s practical (or very costly but acceptable) to pretty strictly cap input bandwidth.

To elaborate on the factors by which the covert input bandwidth cap differs from the covert output bandwidth cap:
Weaker cap: the cap it is practical to place on covert input bandwidth is probably much weaker (i.e. higher) than that on covert output bandwidth—perhaps by a factor 100-100,000, because:
Legitimate input traffic is much larger (maybe 10-100x) even before considering differences in compressibility
Input traffic is likely less compressible than output traffic (maybe by 1-10x); without necessarily being LLM-generated, it’s less amenable to some compression (namely arithmetic coding using an LLM distribution)
Input traffic is likely less feasible to analyze to distinguish legitimate/non-training inputs from training inputs (maybe for a 10-100x difference), since unlike with outputs, we can’t expect randomized recomputation to reproduce results
Compressibility: covert inputs into training (i.e. model weights) are presumably much less compressible than covert outputs (i.e. gradients), because the latter are much sparser. We might guess this is a 100x difference.

(And even if training phased a large delay due to the covert input bandwidth cap, this delay wouldn’t add up with the delay from the covert output bandwidth cap.)
Caps + memory wipes
Memory wipes involve going back and forth between a memory reset phase and an uptime phase. I’ll refer to a memory reset phase followed by an uptime phase as a “memory wipe cycle” or just a “cycle”:


A memory wipe cycle. (There may be an additional “memory offload” phase before the memory wipe.) (Drawn with Excalidraw.)

Here’s a security argument we could make using a covert output/input bandwidth cap + memory wipes:
Covertly outputting a full gradient takes C > 1 memory wipe cycles, due to the covert output bandwidth cap; OR covertly inputting a full covert model takes C > 1 memory wipe cycles, due to the covert input bandwidth cap.
Due to (1), there is some redundancy (we might hope the Prover is forced to redundantly load the covert model C times instead of once, or re-compute the gradient C times instead of once). This greatly reduces training efficiency.

However, I’ll argue that this doesn’t help us much.

First, let’s consider the case where the Prover is only slowed by the cap on covert output bandwidth, not by the cap on covert input bandwidth. Then:
The downtime of each memory wipe will likely be on the order of a minute.
For wiping existing memory, even the relatively quick approach of turning volatile memory off and on (and otherwise dealing with non-volatile memory) may take on the order of a minute, to ensure memory is unrecoverable. 
For reloading legitimate memory, assuming 100 Gbps input and 4 Tb to be inputted (1T FP4 parameters), the time needed would be 4e12/1e11 = 40 seconds.
Given downtime on the order of a minute, then if we don’t want downtime to be worse than ~10%, the uptime in each cycle must be ~10 minutes.
Given uptime that’s comfortably enough to compute gradients (like the above ~10 minutes), the training penalty (i.e. the factor by which covert training is slowed) is at most ~C, the number of cycles needed to covertly output a gradient. So we at least need C = 20 to slow a 3-month training run so that it takes 5 years.
To conservatively achieve C = 20 given an uptime of 10 minutes, we need a covert output bandwidth cap of ~100 Kbps.
But we already knew that a ~100 Kbps covert output cap could, with moderate confidence, slow training to 5 years. So the memory wipes aren’t adding any clear value to the verification.

The above was considering the case where the Prover is only slowed by the cap on covert output bandwidth. Alternatively, if they’re also slowed by the cap on covert input bandwidth, then (by the above arguments) probably they’ll already be prohibitively slowed by the cap on covert output bandwidth. So again, the memory wipes and cap on covert input bandwidth aren’t adding any clear value.
Where memory wipes would be more valuable
There are at least two ways we could make more convenient assumptions, which would make memory wipes ~1-2 OOMs more useful (as in: raise by ~1-2 OOMs the covert input-output bandwidth at which they impose a given training penalty):
Faster wipes: the memory wipe period could maybe be shortened from a ~minute to seconds, but this would require all of the following:
For hardware that’s wiped by being turned off and on, gain assurances, perhaps through empirical testing, that seconds are short enough to prevent cold boot attacks.
For hardware that’s wiped through a cryptographic protocol, gain assurances this is feasible within seconds.
For reloading legitimate memory quickly, substantially increase input bandwidth. (This might be costly even if we’ve given up on detecting what inputs are covert, since input channels may need to be tapped or modified just to ensure they aren’t being used as output channels.)
Higher downtime fraction: Given ~minute-long memory wipe phases, uptime could be ~1-2 OOMs shorter at the enormous cost of increasing downtime from 10% to 50-90%.
Forcing 90% downtime trivially slows down training by 10x, but there could be a further slowdown within the uptime.
Aside: memory wipes for limiting gradient compression?
Separately from the above memory wipe arguments, one thing memory wipes might help with is making it much harder to use “error feedback” in training, which is a technique where one locally accumulates errors that result from gradient compression or other approximations, and feeds this error into the next iteration. Memory wipes along with a covert output bandwidth cap would limit this method, since outputting the error for each gradient term should take about as much bandwidth as outputting the uncompressed gradient, which the Prover is trying to avoid. Limiting this method might reduce how much gradients can be compressed while preserving performance, directly strengthening the “covert output bandwidth cap” argument.
