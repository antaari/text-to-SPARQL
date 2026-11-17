from metrics.mcnemar_test import run_mcnemar_test

results = run_mcnemar_test(
    pairs=[
        ("zs4_phi4", "zs4_pp3_qwen14b_v2"),
        ("fs7_phi4", "fs7_qwen14b"),
        ("cot13_phi4", "cot13_pp3_qwen14b_v2")
    ],
    output_json = "metrics/cross_comparison_phi4xqwen14b.json",
    comparison_file= "global_comparison.json"
)

print(results[0]["metrics"]["correct_queries"]["stats"]["p_value"])