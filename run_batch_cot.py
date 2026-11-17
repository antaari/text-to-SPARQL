from full_pipeline_ollama import run_batch

if __name__ == "__main__":
    run_batch(
        indices=[0],
        generation_type="cot",
        generation_method="cot_v13",
        model_name="qwen2.5-coder:7b",
        preprocessing_method="preprocessing_v12",
        postprocessing_method="postprocessing_v03",
        output_json_path="prova_cot.json",
        method_key="cot13",
        update_comparison=False
    )