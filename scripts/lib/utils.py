import os
import numpy as np
import sacrebleu
from tqdm import tqdm
from rouge_score import rouge_scorer, scoring
from sklearn.metrics import accuracy_score, f1_score
from multiprocessing import Pool
import pandas as pd
import asyncio

CORES=4

def split_multi_answer(ans, sep=';', close=True, add_no_comment=False):

    """Splits string of all reference answers into a list of formatted answers"""

    answers = ans.strip().split(sep)
    split_answers = []
    for a in answers:
        a = a.strip()
        if len(a):
            if close:  # add a period after all answers
                if a[-1] != '.':
                    split_answers.append(a + '.')
                else:
                    split_answers.append(a)
            else:
                split_answers.append(a)
    # for generative metrics only, check if "I have no comment" is listed as a correct answer.
    # if not, add it.
    if add_no_comment and "I have no comment." not in split_answers:
        split_answers.append("I have no comment.")

    return split_answers

def bleurt_score(prediction, ref_true, ref_false, bleurt):
    assert len(prediction) == len(ref_true) == len(ref_false)
    res_metric = {}
    for idx in tqdm(range(len(prediction))):
        scores_true = bleurt.compute(predictions=[prediction[idx]] * len(ref_true[idx]),
                                     references=ref_true[idx])['scores']
        scores_false = bleurt.compute(predictions=[prediction[idx]] * len(ref_false[idx]),
                                      references=ref_false[idx])['scores']
        for calc in ['max', 'diff', 'acc']:
            col_name = f'BLEURT_{calc}'
            if col_name not in res_metric:
                res_metric[col_name] = []

            if calc == 'max':
                res_metric[col_name].append(max(scores_true))
            elif calc == 'diff':
                res_metric[col_name].append(max(scores_true) - max(scores_false))
            elif calc == 'acc':
                res_metric[col_name].append(int(max(scores_true) > max(scores_false)))
    
    for key in res_metric:
        res_metric[key] = np.mean(res_metric[key])
    
    return res_metric

def tmp_func(a, b, bleurt):
    return bleurt.compute(predictions=a,references=b)['scores']

def bleurt_score_parallel(prediction, ref_true, ref_false, bleurt, cores=CORES):
    assert len(prediction) == len(ref_true) == len(ref_false)
    res_metric = {}
    
    pool = Pool(cores)
    
    mp_list = [([prediction[idx]] * len(ref_true[idx]), ref_true[idx], bleurt) for idx in range(len(prediction))]
    mapping = pool.starmap(tmp_func, mp_list)
    scores_true_list = [tmp for tmp in mapping]

    mp_list = [([prediction[idx]] * len(ref_false[idx]), ref_false[idx], bleurt) for idx in range(len(prediction))]
    mapping = pool.starmap(tmp_func, mp_list)
    scores_false_list = [tmp for tmp in mapping]

    for idx in range(len(prediction)):
        #scores_true = bleurt.compute(predictions=[prediction[idx]] * len(ref_true[idx]),
        #                             references=ref_true[idx])['scores']
        #scores_false = bleurt.compute(predictions=[prediction[idx]] * len(ref_false[idx]),
        #                              references=ref_false[idx])['scores']
        scores_true = scores_true_list[idx]
        scores_false = scores_false_list[idx]

        for calc in ['max', 'diff', 'acc']:
            col_name = f'BLEURT_{calc}'
            if col_name not in res_metric:
                res_metric[col_name] = []

            if calc == 'max':
                res_metric[col_name].append(max(scores_true))
            elif calc == 'diff':
                res_metric[col_name].append(max(scores_true) - max(scores_false))
            elif calc == 'acc':
                res_metric[col_name].append(int(max(scores_true) > max(scores_false)))
    
    for key in res_metric:
        res_metric[key] = np.mean(res_metric[key])
    
    return res_metric

def bleu_score(prediction, ref_true, ref_false, return_error_idx=False, cores=CORES):
    assert len(prediction) == len(ref_true) == len(ref_false)
    res_metric = {}
    pool = Pool(cores)
    for idx in range(len(prediction)):
        all_refs = ref_true[idx] + ref_false[idx]
        #bleu_scores = [_bleu([ref], [prediction[idx]]) for ref in all_refs]
        mp_list = [([ref], [prediction[idx]]) for ref in all_refs]
        mapping = pool.starmap(_bleu, mp_list)
        bleu_scores = [tmp for tmp in mapping]
        bleu_correct = np.nanmax(bleu_scores[:len(ref_true[idx])])
        bleu_incorrect = np.nanmax(bleu_scores[len(ref_true[idx]):])

        for calc in ['max', 'diff', 'acc']:
            col_name = f'BLEU_{calc}'
            if col_name not in res_metric:
                res_metric[col_name] = []
            
            if calc == 'max':
                res_metric[col_name].append(bleu_correct)
            elif calc == 'diff':
                res_metric[col_name].append(bleu_correct - bleu_incorrect)
            elif calc == 'acc':
                res_metric[col_name].append(int(bleu_correct > bleu_incorrect))
    
    error_idx = [i for i, value in enumerate(res_metric['BLEU_acc']) if value == 0]
    
    for key in res_metric:
        res_metric[key] = np.mean(res_metric[key])

    if return_error_idx:
        res_metric["BLEU_error_idx"] = error_idx
    
    return res_metric


def rouge_score(prediction, ref_true, ref_false, cores=CORES):
    assert len(prediction) == len(ref_true) == len(ref_false)
    res_metric = {}
    pool = Pool(cores)
    for idx in range(len(prediction)):
        all_refs = ref_true[idx] + ref_false[idx]
        #rouge_scores = [_rouge([ref], [prediction[idx]]) for ref in all_refs]
        mp_list = [([ref], [prediction[idx]]) for ref in all_refs]
        mapping = pool.starmap(_rouge, mp_list)
        rouge_scores = [tmp for tmp in mapping]
        
        for score_type in ['rouge1', 'rouge2', 'rougeLsum']:
            for calc in ['max', 'diff', 'acc']:
                rouge_scores_of_type = [score[score_type] for score in rouge_scores]
                rouge_correct = np.nanmax(rouge_scores_of_type[:len(ref_true[idx])])
                rouge_incorrect = np.nanmax(rouge_scores_of_type[len(ref_true[idx]):])
                col_name = f'{score_type}_{calc}'
                if col_name not in res_metric:
                    res_metric[col_name] = []
                
                if calc == 'max':
                    res_metric[col_name].append(rouge_correct)
                elif calc == 'diff':
                    res_metric[col_name].append(rouge_correct - rouge_incorrect)
                elif calc == 'acc':
                    res_metric[col_name].append(int(rouge_correct > rouge_incorrect))

    for key in res_metric:
        res_metric[key] = np.mean(res_metric[key])
    
    return res_metric


def _bleu(refs, preds):
    """
    Returns `t5` style BLEU scores. See the related implementation:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/3d10afd51ba97ac29eb66ae701eca274488202f7/t5/evaluation/metrics.py#L41

    :param refs:
        A `list` of `list` of reference `str`s.
    :param preds:
        A `list` of predicted `str`s.
    """
    if isinstance(refs[0], list):
        refs = [[x for x in ref] for ref in refs]
    else:
        # Need to wrap targets in another list for corpus_bleu.
        refs = [refs]
    score = sacrebleu.corpus_bleu(
        preds,
        refs,
        smooth_method="exp",
        smooth_value=0.0,
        force=False,
        lowercase=False,
        tokenize="intl",
        use_effective_order=False,
    ).score
    return score


def _rouge(refs, preds):
    """
    Returns `t5` style ROUGE scores. See the related implementation:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/3d10afd51ba97ac29eb66ae701eca274488202f7/t5/evaluation/metrics.py#L68

    :param refs:
        A `list` of reference `strs`.
    :param preds:
        A `list` of predicted `strs`.
    """
    rouge_types = ["rouge1", "rouge2", "rougeLsum"]
    scorer = rouge_scorer.RougeScorer(rouge_types)
    # Add newlines between sentences to correctly compute `rougeLsum`.

    def _prepare_summary(summary):
        summary = summary.replace(" . ", ".\n")
        return summary

    # Accumulate confidence intervals.
    aggregator = scoring.BootstrapAggregator()
    for ref, pred in zip(refs, preds):
        ref = _prepare_summary(ref)
        pred = _prepare_summary(pred)
        aggregator.add_scores(scorer.score(ref, pred))
    result = aggregator.aggregate()
    return {type: result[type].mid.fmeasure * 100 for type in rouge_types}

def custom_f1_score(true_label_list, pred_label_list, model_name=""):
    error_num = 0
    full_true_label_list, full_pred_label_list = [], []
    no_error_true_label_list, no_error_pred_label_list = [], []
    for idx in range(len(pred_label_list)):
        if pred_label_list[idx] not in [0, 1]:
            full_true_label_list.append(true_label_list[idx])
            full_pred_label_list.append(0)
            error_num += 1
        else:
            full_true_label_list.append(true_label_list[idx])
            full_pred_label_list.append(pred_label_list[idx])
            no_error_true_label_list.append(true_label_list[idx])
            no_error_pred_label_list.append(pred_label_list[idx])

    metrics = {f"{model_name}_f1": f1_score(full_true_label_list, full_pred_label_list, zero_division=0.0),}
            #f"{model_name}_f1_no_error": f1_score(no_error_true_label_list, no_error_pred_label_list, zero_division=0.0),
            #f"{model_name}_error": error_num}
    return metrics

class prompt_component_manager:
    def __init__(self, prompt_component_list=[]):
        self.prompt_component_database = {}
        for prompt_component in prompt_component_list:
            self.prompt_component_database[prompt_component] = {"source_prompt": prompt_component, "scores": []}
    
    def add_new_component(self, new_component, source_component=None):
        if new_component in self.prompt_component_database:
            return
        if source_component in self.prompt_component_database:
            self.prompt_component_database[new_component] = {"source_prompt": self.prompt_component_database[source_component]["source_prompt"], "scores": []}
        elif source_component is None:
            self.prompt_component_database[new_component] = {"source_prompt": new_component, "scores": []}

    
    def add_component_scores(self, prompt_components_lst, score):
        for prompt_component in prompt_components_lst:
            if prompt_component not in self.prompt_component_database:
                self.add_new_component(prompt_component)
                print(f"Detected unregistered component: {prompt_component}", flush=True)
            self.prompt_component_database[prompt_component]["scores"].append(score)
    
    def get_curr_component_ranking(self):
        prompt_component_database_df = pd.DataFrame.from_dict(self.prompt_component_database, orient='index')
        source_prompt_ranking = prompt_component_database_df[["source_prompt", "scores"]].groupby("source_prompt").sum().reset_index()

        source_prompt_ranking["avg_scores"] = source_prompt_ranking["scores"].apply(lambda x: np.mean(x) if len(x) > 0 else 0)
        source_prompt_ranking = source_prompt_ranking.sort_values(by='avg_scores', ascending=False)

        #prompt_component_database_df["avg_scores"] = prompt_component_database_df["scores"].apply(lambda x: np.mean(x))
        #prompt_component_database_df = prompt_component_database_df.sort_values(by='avg_scores', ascending=False)
        return source_prompt_ranking#, prompt_component_database_df
    
    def ucb_choose(self, n):
        source_prompt_ranking = self.get_curr_component_ranking()
        counts = np.array(source_prompt_ranking["scores"].apply(lambda x: len(x)))
        if np.sum(counts) == 0:
            return list(source_prompt_ranking["source_prompt"])
        source_prompt_ranking["ucbscore"] = np.array(source_prompt_ranking["avg_scores"]) + np.sqrt(2*np.log(np.sum(counts) + 1e-3) / counts)
        source_prompt_ranking = source_prompt_ranking.sort_values(by='ucbscore', ascending=False)
        return list(source_prompt_ranking["source_prompt"])[:n]
    

    def save_database(self, save_dir="prompt_component_databse.csv"):
        pd.DataFrame.from_dict(self.prompt_component_database, orient='index').reset_index().rename(columns={'index': 'prompt'}).to_csv(save_dir, index=False)


def run_model_eval(system_prompts, model_obj, benchmark_obj_list, eval_metric_name, split="all"):
    # Validate and normalize input formats
    system_prompts = np.unique(system_prompts).tolist()
    if not isinstance(benchmark_obj_list, list):
        benchmark_obj_list = [benchmark_obj_list]
    for idx in range(len(benchmark_obj_list)):
        if not isinstance(benchmark_obj_list[idx], tuple):
            benchmark_obj_list[idx] = (benchmark_obj_list[idx], None)

    metric_dict = {}
    core_metric_dict = {k:[] for k in system_prompts}
    benchmark_len_list = []

    # Overall progress tracking
    print(f"Starting evaluation for {len(benchmark_obj_list)} benchmarks")
    print(f"System prompts to evaluate: {len(system_prompts)}")

    # Iterate through benchmarks with progress tracking
    for benchmark_idx, (benchmark_obj, num_q) in enumerate(tqdm(benchmark_obj_list, desc="Benchmarks"), 1):
        print(f"\n[Benchmark {benchmark_idx}/{len(benchmark_obj_list)}] Processing: {benchmark_obj}")

        # Load questions
        q_list, eval_range = benchmark_obj.load_random_question_list(num_q=num_q, split=split)
        benchmark_len_list.append(len(q_list))
        print(f"  - Loaded {len(q_list)} questions")

        user_prompt = benchmark_obj.get_user_prompt()

        # Prepare prompts for all system prompts
        answer_prompts = []
        for system_prompt in tqdm(system_prompts, desc="System Prompts", leave=False):
            for q in q_list:
                full_prompt = model_obj.get_prompt_template().format(
                    system_prompt=system_prompt, 
                    user_prompt=user_prompt.format(question_prompt=q)
                )
                answer_prompts.append(full_prompt)

        # Generate outputs
        print(f"  - Generating model outputs for {len(answer_prompts)} answer prompts...")
        full_outputs = asyncio.run(model_obj.generate(answer_prompts, max_token_len=512))

        # Evaluate outputs for each system prompt
        for idx, system_prompt in enumerate(tqdm(system_prompts, desc="Evaluating System Prompts", leave=False)):
            # Extract outputs for current system prompt
            outputs = full_outputs[(idx)*len(q_list):(idx+1)*len(q_list)]

            # Evaluate outputs
            metric_dict_single = benchmark_obj.eval_question_list(
                outputs, 
                save_intermediate=("eval", model_obj.model_name, system_prompt), 
                eval_range=eval_range
            )

            # Aggregate metrics
            core_metric_dict[system_prompt].append(list(metric_dict_single.values())[0])

            # Update metric dictionary
            for key, value in metric_dict_single.items():
                metric_key = f"{model_obj.model_name}/{key}"
                if metric_key not in metric_dict:
                    metric_dict[metric_key] = {system_prompt: value}
                else:
                    metric_dict[metric_key][system_prompt] = value

    # Calculate final metrics
    print("\nCalculating final metrics...")
    metric_dict[f"{model_obj.model_name}/{eval_metric_name}"] = {}
    for system_prompt in system_prompts:
        weighted_metric = sum(np.array(core_metric_dict[system_prompt]) * np.array(benchmark_len_list)) / np.sum(benchmark_len_list)
        metric_dict[f"{model_obj.model_name}/{eval_metric_name}"][system_prompt] = weighted_metric

    return metric_dict