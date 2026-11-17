from full_pipeline_ollama import run_batch

if __name__ == "__main__":
    run_batch(
        indices=[0],
        generation_type="zero-shot",
        generation_method="zero_shot_v04",
        model_name="qwen2.5-coder:7b",
        preprocessing_method="preprocessing_v12",
        postprocessing_method="postprocessing_v03",
        output_json_path="prova_zeroshot.json",
        method_key="zs4",
        update_comparison=False
    )