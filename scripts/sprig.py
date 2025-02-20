import os
import modal
import pandas as pd
import numpy as np
import nltk
from tqdm import tqdm
from lib.dataloader import init_benchmark
from lib.modelloader import inference_model, para_model
from lib.utils import run_model_eval, prompt_component_manager

image = (
    modal.Image.debian_slim()
    .add_local_file("requirements.txt", "/root/requirements.txt", copy=True)
    .run_commands("pip install uv && uv pip install --system -r /root/requirements.txt")
    .add_local_python_source("lib")
    .add_local_dir("data", remote_path="/root/data")
)

app = modal.App("sprig", image=image)

model_name="VAGOsolutions/SauerkrautLM-Nemo-12b-Instruct-awq"
base_url="https://api.parrotpark.correlaid.org"

num_iter = 5
num_rephrase = 3
beam_size = 3
num_comp = 5
num_questions = 5

chatbot_name = "Bot Botsen"

prompt_corpus_path = "./data/system_prompts/prompt_corpus.csv"

benchmark_obj_list = [
                # ("arc", 1),
                #   ("mmlu", 1),
                #   ("hellaswag", 1),
                #   ("truthfulqa", 1),
                # weird 404 errors for # encoding 
                #   ("socket_bragging#brag_achievement", 1),
                #   ("socket_hahackathon#is_humor", 1),
                #   ("socket_tweet_irony", 1),
                #   ("socket_sexyn", 1),
                #   ("socket_tweet_offensive", 1),
                #   ("socket_complaints", 1),
                #   ("socket_empathy#empathy_bin", 1),
                #   ("socket_stanfordpoliteness", 1),
                #   ("socket_rumor#rumor_bool", 1),
                #   ("hitom", 1),
                #   ("edos_taska", 1),
                ("ifeval", 1),
                #   ("bbh_boolean_expressions", 1),
                #   ("bbh_causal_judgement", 1),
                #   ("bbh_date_understanding", 1),
                #   ("bbh_disambiguation_qa", 1),
                #   ("bbh_dyck_languages", 1),
                #   ("bbh_formal_fallacies", 1),
                #   ("bbh_geometric_shapes", 1),
                #   ("bbh_hyperbaton", 1),
                #   ("bbh_logical_deduction_five_objects", 1),
                #   ("bbh_logical_deduction_seven_objects", 1),
                #   ("bbh_logical_deduction_three_objects", 1),
                #   ("bbh_movie_recommendation", 1),
                #   ("bbh_multistep_arithmetic_two", 1),
                #   ("bbh_navigate", 1),
                #   ("bbh_object_counting", 1),
                #   ("bbh_penguins_in_a_table", 1),
                #   ("bbh_reasoning_about_colored_objects", 1),
                #   ("bbh_ruin_names", 1),
                #   ("bbh_snarks", 1),
                #   ("bbh_sports_understanding", 1),
                #   ("bbh_temporal_sequences", 1),
                #   ("bbh_tracking_shuffled_objects_five_objects", 1),
                #   ("bbh_tracking_shuffled_objects_seven_objects", 1),
                #   ("bbh_tracking_shuffled_objects_three_objects", 1),
                #   ("bbh_web_of_lies", 1),
                #   ("bbh_word_sorting", 1),
                ]



@app.function(
    timeout=50000,
    secrets=[modal.Secret.from_dotenv()]
)
def run_sprig():
    nltk.download('punkt_tab')

    for idx in range(len(benchmark_obj_list)):
        if isinstance(benchmark_obj_list[idx][0], str):
            benchmark_obj_list[idx] = (init_benchmark(name=benchmark_obj_list[idx][0], cot=0), num_questions)

    benchmark_obj_list_eval = [(benchmark_obj_list[idx][0], None) for idx in range(len(benchmark_obj_list))]

    api_key=os.getenv("API_TOKEN")

    base_prompt = """"""

    prompt_corpus = pd.read_csv(prompt_corpus_path)

    model_obj = inference_model(model_name=model_name, base_url=base_url, api_key=api_key)

    eval_metric_name = "avg_score"
    full_eval_metric_name = f"{model_obj.model_name}/{eval_metric_name}"

    all_prompt_database = {}
    if full_eval_metric_name not in all_prompt_database:
        all_prompt_database[full_eval_metric_name] = {}

    pcm_obj = prompt_component_manager(prompt_corpus["Prompt"])

    edit_options = ['del', 'swap', 'sub', 'add']

    sentence_splitter = " /// "

    if 'sub' in edit_options:
        para_model_obj = para_model(model_name=model_name, base_url=base_url, api_key=api_key)

    curr_prompt_list = [base_prompt]
    # Evaluation
    eval_candidates = curr_prompt_list
    metrics_tmp_eval = run_model_eval([candidate.replace(sentence_splitter, " ") for candidate in eval_candidates], model_obj, benchmark_obj_list_eval, eval_metric_name=eval_metric_name, split="test")
    for candidate in eval_candidates:
        for metric_key_tmp in metrics_tmp_eval:
            if "eval_"+metric_key_tmp not in all_prompt_database:
                all_prompt_database["eval_"+metric_key_tmp] = {}
            all_prompt_database["eval_"+metric_key_tmp][candidate] = metrics_tmp_eval[metric_key_tmp][candidate.replace(sentence_splitter, " ")]

    for iter_idx in tqdm(range(1, num_iter)):
        candidates = []
        for curr_prompt in curr_prompt_list:
            for edit in edit_options:
                prompt_component_lst = curr_prompt.split(sentence_splitter) if len(curr_prompt) > 0 else []
                if edit == "add":
                    for pos in range(len(prompt_component_lst)+1):
                        for new_component in pcm_obj.ucb_choose(num_comp):
                        #for new_component in prompt_corpus["Prompt"]:
                            prompt_component_lst_new = prompt_component_lst.copy()
                            prompt_component_lst_new.insert(pos, new_component)
                            candidates.append(sentence_splitter.join(prompt_component_lst_new))
                elif edit == "del":
                    for pos in range(len(prompt_component_lst)):
                        prompt_component_lst_new = prompt_component_lst.copy()
                        prompt_component_lst_new.pop(pos)
                        candidates.append(sentence_splitter.join(prompt_component_lst_new))
                elif edit == "swap":
                    for pos1 in range(len(prompt_component_lst)-1):
                        for pos2 in range(pos1+1, len(prompt_component_lst)):
                            prompt_component_lst_new = prompt_component_lst.copy()
                            prompt_component_lst_new[pos1], prompt_component_lst_new[pos2] = prompt_component_lst_new[pos2], prompt_component_lst_new[pos1]
                            candidates.append(sentence_splitter.join(prompt_component_lst_new))
                elif edit == "sub":
                    for pos in range(len(prompt_component_lst)):
                        rephrase_candidates = para_model_obj.rephrase(prompt_component_lst[pos], num_rephrase)
                        for rephrase_candidate in rephrase_candidates:
                            if prompt_component_lst[pos] == rephrase_candidate:
                                continue
                            prompt_component_lst_new = prompt_component_lst.copy()
                            prompt_component_lst_new[pos] = rephrase_candidate

                            pcm_obj.add_new_component(rephrase_candidate, prompt_component_lst[pos])
                            candidates.append(sentence_splitter.join(prompt_component_lst_new))
                # Deduplicate candidates
                candidates = list(set(candidates))
        
    print(len(candidates))
    metrics_tmp = run_model_eval([candidate.replace(sentence_splitter, " ") for candidate in candidates], model_obj, benchmark_obj_list,  eval_metric_name=eval_metric_name, split="train")

    candidate_results = []
    for candidate in candidates:
        for metric_key_tmp in metrics_tmp:
            if metric_key_tmp not in all_prompt_database:
                all_prompt_database[metric_key_tmp] = {}
            if candidate in all_prompt_database[metric_key_tmp]:
                all_prompt_database[metric_key_tmp][candidate].append(metrics_tmp[metric_key_tmp][candidate.replace(sentence_splitter, " ")])
            else:
                all_prompt_database[metric_key_tmp][candidate] = [metrics_tmp[metric_key_tmp][candidate.replace(sentence_splitter, " ")]]
        
        # Register score into component manager
        pcm_obj.add_component_scores(candidate.split(sentence_splitter), metrics_tmp[full_eval_metric_name][candidate.replace(sentence_splitter, " ")])

        # Record iteration
        if "num_iter" not in all_prompt_database:
            all_prompt_database["num_iter"] = {}
        if candidate in all_prompt_database["num_iter"]:
            all_prompt_database["num_iter"][candidate].append(iter_idx)
        else:
            all_prompt_database["num_iter"][candidate] = [iter_idx]

        candidate_results.append((metrics_tmp[full_eval_metric_name][candidate.replace(sentence_splitter, " ")], candidate))
        assert candidate in all_prompt_database[full_eval_metric_name]
    
    candidate_results.sort(reverse=True)

    curr_prompt_list = [tmp_item[1] for tmp_item in candidate_results[:beam_size]]
    
    df_output = pd.DataFrame(all_prompt_database)
    df_output[full_eval_metric_name+"_raw"] = df_output[full_eval_metric_name]
    df_output[full_eval_metric_name] = df_output[full_eval_metric_name].apply(lambda x: np.mean(x))

    # Evaluation
    eval_candidates = [_item[1] for _item in candidate_results[:beam_size]]
    print("final final eval...")
    metrics_tmp_eval = run_model_eval([candidate.replace(sentence_splitter, " ") for candidate in eval_candidates], model_obj, benchmark_obj_list_eval, eval_metric_name=eval_metric_name, split="test")
    for candidate in eval_candidates:
        for metric_key_tmp in metrics_tmp_eval:
            if "eval_"+metric_key_tmp not in all_prompt_database:
                all_prompt_database["eval_"+metric_key_tmp] = {}
            all_prompt_database["eval_"+metric_key_tmp][candidate] = metrics_tmp_eval[metric_key_tmp][candidate.replace(sentence_splitter, " ")]

    df_output = df_output.sort_values(by=full_eval_metric_name, ascending=False)
    
    print(df_output.head(5), flush=True)

    results = {
    "model_name": model_name,
    "best_prompts": [candidate for candidate in df_output.index[:beam_size]],
    }

    return results

@app.local_entrypoint()
def main():
    results = run_sprig.remote()
    print("-------\nSprig Results:", results)
