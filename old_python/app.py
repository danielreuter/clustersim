"""
Streamlit UI for the Erdil throughput model.
Run with: streamlit run app.py
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from erdil import (
    maximize_cluster_throughput,
    MODEL_MAP,
    GPU_MAP,
)

st.set_page_config(page_title="Inference Throughput Model", layout="wide")
st.title("Inference Throughput Model")
st.caption("Based on Erdil 2025 (arXiv:2506.04645)")

# =============================================================================
# Sidebar Controls
# =============================================================================
st.sidebar.header("Model & Hardware")

model_name = st.sidebar.selectbox("Model", list(MODEL_MAP.keys()), index=0)
gpu_name = st.sidebar.selectbox("GPU", list(GPU_MAP.keys()), index=0)

model = MODEL_MAP[model_name]
gpu = GPU_MAP[gpu_name]

st.sidebar.header("Cluster")

# Log slider for cluster size
log_cluster = st.sidebar.slider(
    "Cluster Size (log₁₀ GPUs)",
    0.0, 5.0, 4.0, 0.1,
    help="10^x GPUs. 4.0 = 10,000 GPUs"
)
cluster_gpus = int(round(10 ** log_cluster))
st.sidebar.caption(f"Cluster: **{cluster_gpus:,}** GPUs")

st.sidebar.header("Context")

context_options = [0, 256, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
context_len = st.sidebar.select_slider(
    "Context Length (tokens)",
    options=context_options,
    value=4096,
    help="Input/prompt length in tokens"
)

st.sidebar.header("Resources")

# Log slider for HBM fraction
log_hbm = st.sidebar.slider(
    "HBM Fraction (log₁₀)",
    -3.0, 0.0, 0.0, 0.1,
    help="10^x fraction. 0.0 = 100%, -1.0 = 10%, -2.0 = 1%"
)
hbm_fraction = 10 ** log_hbm
st.sidebar.caption(f"HBM: **{hbm_fraction*100:.1f}%** ({gpu.hbm_size_bytes * hbm_fraction / 1e9:.1f} GB/GPU)")

gamma_overlap = st.sidebar.slider(
    "γ (Comm Overlap)",
    0.0, 1.0, 0.5, 0.1,
    help="0 = no overlap (comm after compute), 1 = full overlap (comm during compute)"
)

st.sidebar.header("Advanced")

use_pp = st.sidebar.checkbox("Enable Pipeline Parallelism", value=True)

with st.sidebar.expander("Latency Model (Erdil 2025)"):
    st.markdown("""
**NCCL-calibrated latencies** (H100 SXM):
- Kernel launch: 4 µs/op
- NVLink: ~1 µs/step
- IB: ~5-10 µs/step

**Overlap model (γ)**:
```
t_no_overlap = max(compute,mem) + comm_lat + comm_bw
t_full_overlap = max(compute,mem, comm_bw) + comm_lat
t_actual = (1-γ)*t_no + γ*t_full
```

**Key insight**: At scale, latency often dominates BW.
    """)

# =============================================================================
# Compute Throughput
# =============================================================================

@st.cache_data
def compute_throughput(_model_name, _gpu_name, _cluster_gpus, _context_len, _hbm_fraction, _gamma, _use_pp):
    """Cached throughput computation."""
    m = MODEL_MAP[_model_name]
    g = GPU_MAP[_gpu_name]
    return maximize_cluster_throughput(
        cluster_gpus=_cluster_gpus,
        model=m,
        gpu=g,
        input_len=_context_len,
        hbm_fraction=_hbm_fraction,
        gamma_overlap=_gamma,
        use_pp=_use_pp,
    )

throughput, info = compute_throughput(
    model_name, gpu_name, cluster_gpus, context_len, hbm_fraction, gamma_overlap, use_pp
)

# =============================================================================
# Display Results
# =============================================================================

if "error" in info:
    st.error(f"No valid configuration: {info['error']}")
    st.info("Try increasing HBM fraction or cluster size.")
else:
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Throughput", f"{throughput/1e6:.2f} M tok/s")
    col2.metric("Efficiency", f"{info.get('efficiency', 0)*100:.1f}%")
    col3.metric("Config", f"PP={info.get('pp', 1)}, TP={info.get('tp', 1)}")
    col4.metric("Bottleneck", info.get('bottleneck', 'N/A'))

    # Second row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("GPUs/Instance", info.get('gpus_per_instance', 1))
    col2.metric("Replicas", info.get('replicas', 1))
    col3.metric("Batch Size", info.get('batch', 1))
    col4.metric("2D TP", "Yes" if info.get('two_d_tp', False) else "No")

    # Time breakdown
    st.subheader("Time Breakdown (per token)")

    t_compute = info.get('t_compute_s', 0) * 1e3  # ms
    t_mem = info.get('t_mem_s', 0) * 1e3
    t_comm_lat = info.get('t_comm_lat_s', 0) * 1e3
    t_comm_bw = info.get('t_comm_bw_s', 0) * 1e3
    t_total = info.get('t_total_s', 0) * 1e3

    t_comm = t_comm_lat + t_comm_bw
    t_compute_mem = max(t_compute, t_mem)

    # Display breakdown
    lat_frac = t_comm_lat / t_comm * 100 if t_comm > 0 else 0
    st.markdown(f"""
**Raw times** (before overlap):
- **Compute**: {t_compute:.3f} ms — tensor core FLOPs + kernel launch
- **Memory**: {t_mem:.3f} ms — HBM reads (weights + KV cache)
- **Comm**: {t_comm:.3f} ms — **{t_comm_lat:.3f} latency** ({lat_frac:.0f}%) + {t_comm_bw:.3f} bandwidth

**Overlap model** (γ = {gamma_overlap:.1f}):
```
wall_time = (1-γ)*(max(comp,mem) + comm) + γ*(max(comp,mem,comm_bw) + comm_lat)
          = {t_total:.3f} ms
```
    """)

    # Gantt-style chart
    fig, ax = plt.subplots(figsize=(10, 2.5))

    # Calculate positions based on overlap model
    comm_bw_start = (1 - gamma_overlap) * t_compute_mem
    comm_bw_end = comm_bw_start + t_comm_bw
    comm_lat_start = max(t_compute_mem, comm_bw_end)

    rows = {'Compute': 3, 'Memory': 2, 'Comm β': 1, 'Comm α': 0}
    bar_height = 0.7

    # Draw bars
    if t_compute > 0:
        ax.barh(rows['Compute'], t_compute, left=0, height=bar_height,
                color='#3498db', edgecolor='black', linewidth=0.5, label='Compute')
    if t_mem > 0:
        ax.barh(rows['Memory'], t_mem, left=0, height=bar_height,
                color='#2ecc71', edgecolor='black', linewidth=0.5, label='Memory')
    if t_comm_bw > 0:
        ax.barh(rows['Comm β'], t_comm_bw, left=comm_bw_start, height=bar_height,
                color='#f39c12', edgecolor='black', linewidth=0.5, label='Comm β (bw)')
    if t_comm_lat > 0:
        ax.barh(rows['Comm α'], t_comm_lat, left=comm_lat_start, height=bar_height,
                color='#e74c3c', edgecolor='black', linewidth=0.5, label='Comm α (lat)')

    # Overlap region
    if gamma_overlap > 0 and t_comm_bw > 0:
        overlap_end = min(t_compute_mem, comm_bw_end)
        if overlap_end > comm_bw_start:
            ax.axvspan(comm_bw_start, overlap_end, alpha=0.15, color='gray',
                      hatch='///', label='Overlap region')

    # Wall time marker
    ax.axvline(t_total, color='black', linestyle='--', linewidth=2.5, label=f'Wall: {t_total:.2f}ms')
    ax.axvline(t_compute_mem, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)

    ax.set_yticks(list(rows.values()))
    ax.set_yticklabels(list(rows.keys()))
    ax.set_xlabel("Time (ms)")
    ax.set_xlim(0, t_total * 1.15)
    ax.set_ylim(-0.5, 3.5)
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(loc='upper right', fontsize=7, ncol=2)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.caption(f"**Bottleneck: {info.get('bottleneck', 'N/A')}** — Compute/Memory compete (take max), Comm β overlaps by γ={gamma_overlap:.0%}, Comm α always serial")

    # ==========================================================================
    # Sensitivity Analysis
    # ==========================================================================
    st.subheader("Sensitivity Analysis")

    sens_metric = st.selectbox(
        "Outcome variable",
        ["Efficiency %", "Throughput (M tok/s)", "PP", "TP", "Batch"]
    )

    def extract_metric(tp, inf, metric):
        if "error" in inf:
            return 0
        if metric == "Efficiency %":
            return inf.get('efficiency', 0) * 100
        elif metric == "Throughput (M tok/s)":
            return tp / 1e6
        elif metric == "PP":
            return inf.get('pp', 0)
        elif metric == "TP":
            return inf.get('tp', 0)
        elif metric == "Batch":
            return inf.get('batch', 0)
        return 0

    current_val = extract_metric(throughput, info, sens_metric)

    # Sweep ranges
    hbm_range = np.array([-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0])
    cluster_range = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    context_range = np.array([0, 256, 1024, 4096, 8192, 16384, 32768, 65536])
    gamma_range = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    fig, axes = plt.subplots(1, 4, figsize=(14, 3))

    # 1. HBM sweep
    hbm_vals = []
    for h in hbm_range:
        tp, inf = compute_throughput(model_name, gpu_name, cluster_gpus, context_len, 10**h, gamma_overlap, use_pp)
        hbm_vals.append(extract_metric(tp, inf, sens_metric))

    axes[0].plot([10**h * 100 for h in hbm_range], hbm_vals, '#2ecc71', linewidth=2)
    axes[0].axvline(hbm_fraction * 100, color='red', linestyle='--', linewidth=2)
    axes[0].scatter([hbm_fraction * 100], [current_val], color='red', s=100, zorder=5)
    axes[0].set_xlabel("HBM Fraction %")
    axes[0].set_title("HBM Fraction")
    axes[0].set_xscale('log')
    axes[0].grid(True, alpha=0.3)

    # 2. Cluster size sweep
    cluster_vals = []
    for c in cluster_range:
        tp, inf = compute_throughput(model_name, gpu_name, int(10**c), context_len, hbm_fraction, gamma_overlap, use_pp)
        cluster_vals.append(extract_metric(tp, inf, sens_metric))

    axes[1].plot([10**c for c in cluster_range], cluster_vals, '#3498db', linewidth=2)
    axes[1].axvline(cluster_gpus, color='red', linestyle='--', linewidth=2)
    axes[1].scatter([cluster_gpus], [current_val], color='red', s=100, zorder=5)
    axes[1].set_xlabel("Cluster GPUs")
    axes[1].set_title("Cluster Size")
    axes[1].set_xscale('log')
    axes[1].grid(True, alpha=0.3)

    # 3. Context length sweep
    ctx_vals = []
    for c in context_range:
        tp, inf = compute_throughput(model_name, gpu_name, cluster_gpus, int(c), hbm_fraction, gamma_overlap, use_pp)
        ctx_vals.append(extract_metric(tp, inf, sens_metric))

    axes[2].plot(context_range, ctx_vals, '#9b59b6', linewidth=2)
    axes[2].axvline(context_len, color='red', linestyle='--', linewidth=2)
    axes[2].scatter([context_len], [current_val], color='red', s=100, zorder=5)
    axes[2].set_xlabel("Context Length")
    axes[2].set_title("Context Length")
    axes[2].set_xscale('log')
    axes[2].grid(True, alpha=0.3)

    # 4. Gamma overlap sweep
    gamma_vals = []
    for g in gamma_range:
        tp, inf = compute_throughput(model_name, gpu_name, cluster_gpus, context_len, hbm_fraction, g, use_pp)
        gamma_vals.append(extract_metric(tp, inf, sens_metric))

    axes[3].plot(gamma_range, gamma_vals, '#e74c3c', linewidth=2)
    axes[3].axvline(gamma_overlap, color='red', linestyle='--', linewidth=2)
    axes[3].scatter([gamma_overlap], [current_val], color='red', s=100, zorder=5)
    axes[3].set_xlabel("γ (Overlap)")
    axes[3].set_title("Comm Overlap")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Model info
    with st.expander("Model Details"):
        st.markdown(f"""
**{model_name}**
- Total params: {model.total_params/1e9:.1f}B
- Active params: {model.total_active_params/1e9:.1f}B
- Layers: {model.layers}
- d_model: {model.d_model}
- d_ff: {model.d_ff}
- Experts: {model.n_experts} ({model.n_active_experts} active)
- KV cache/token: {model.kv_cache_size_per_input_bytes/1e3:.1f} KB

**{gpu_name}**
- HBM: {gpu.hbm_size_bytes/1e9:.0f} GB
- HBM BW: {gpu.hbm_bandwidth_Bps/1e12:.1f} TB/s
- FLOPs (BF16): {gpu.flop_per_second.get(16, 0)/1e15:.1f} PFLOP/s
- Node size: {gpu.node_size} GPUs
        """)
