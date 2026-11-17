from metrics.plot_error_histograms import plot_error_histograms

# Dichiarazione esplicita dei file target
target_files = {
    "Zero-shot": "zs4_pp3_qwen14b_v2.json",
    "Few-shot": "fs7_qwen14b.json",
    "Chain-of-Thoughts": "cot13_pp3_qwen14b_v2.json"
}

# Invocazione della funzione
results = plot_error_histograms(
    target_files=target_files,
    output_dir="metrics/error_analysis",
    output_prefix = "zsVSfsVScot_14b"
)

