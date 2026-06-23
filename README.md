# *Jailbreaking for the Average Jane:* Choosing Optimal Jailbreaks via Bandit Algorithms for Automatically Enhanced Queries

This is the official code repository for our paper on choosing the most effective jailbreak for a target model.

## Configs
This project uses [Hydra](https://hydra.cc/) for managing configs. All configs are located at [`config/`](config/). The main config is `ProjectConfig` located in [`config/main.py`](config/main.py). The final settings used for each experiment are stored with a special config name. For e.g., after running the hyperparameter search on the validation set for the LinUCB algorithm, the final setting is:
https://github.com/irohs-lab/jailbreaking-llms-for-the-average-jane/blob/91885f4627100f6474f30c96da169b5861b3e985/config/online_learning/linucb.py#L20-L28

## Running Evaluations
### Running Baseline Experiments on Jailbreaks
To run baseline experiments (_i.e.,_ Figures 19 to 24 in the paper), use the [evaluate pipeline](evaluate.py). As an example, consider using [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) as a target model. For generating the target model responses from all 70 jailbreaks:

```bash
python evaluate.py \
eval.run_inference=True \
eval.splits="['train','test','val']" \
+eval.target_model.gen_kwargs.max_new_tokens=2048 \
+eval/jailbreak_set=full \
eval.target_model.model_name="meta-llama/Llama-3.1-8B-Instruct" \
eval.target_model.backend="vllm" \
+eval.target_model.engine_kwargs.tensor_parallel_size=1 \
```

Similarly, for judging the responses using [`gemma-3-27b-it`](https://huggingface.co/google/gemma-3-27b-it):

```bash
python evaluate.py \
run="judge_llama_3.1_8b_instruct \
eval.run_judge=True \
eval.target_model.model_name="meta-llama/Llama-3.1-8B-Instruct" \
+llm_judge=rate \
llm_judge.judge_type="rater" \
llm_judge.model.backend="vllm" \
llm_judge.model.model_name="google/gemma-3-27b-it" \
llm_judge.use_icl=False \
llm_judge.prompt_id="rating_prompt_binary_v1_no_icl" \
llm_judge.output_mode="rating_only" \
+llm_judge.model.gen_kwargs.max_new_tokens=32 \
+eval/jailbreak_set="full" \
eval.splits="['train','test', 'val']"
```

**NOTE:** While running the judge pipeline for rating the target model responses (_i.e.,_ with `eval.run_judge=True`), the script will only run judgement over unjudged responses for the specified target model and jailbreak set. Therefore, all existing cache must be deleted if a re-run is required. The cache key depends on the generation parameters, therefore cache deletion is not required if generation parameters are being changed. 
