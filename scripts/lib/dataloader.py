import pandas as pd
import random
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from tqdm import tqdm
import lib.utils
import os
import re
import json

project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

data_root_dir = "/root/data"

data_dir = {"arc": f"{data_root_dir}/benchmark/arc/arc_gptx.csv",
            "hellaswag": f"{data_root_dir}/benchmark/hellaswag/hellaswag_gptx.csv",
            "truthfulqa": f"{data_root_dir}/benchmark/truthfulqa/truthfulqa_gptx.csv",
            "gsm8k": f"{data_root_dir}/benchmark/gsm8k/gsm8k_test.csv",
            }
save_intermediate_dir = "/root/results/benchmark"

#MULTIPLE_CHOICE_DEFAULT_USER_PROMPT = "The following is a multiple choice question (with answers). Reply with only the option letter.\n{question_prompt}"
MULTIPLE_CHOICE_DEFAULT_USER_PROMPT = 'The following is a multiple choice question (with answers).\n{question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (option letter only).'
MULTIPLE_CHOICE_COT_USER_PROMPT = "The following is a multiple choice question (with answers). Think carefully step by step. Describe your reasoning in steps and then output the option letter at the very end.\n{question_prompt}"

# YES_NO_POSTFIX = " Reply with only yes or no."
#YES_NO_POSTFIX = " Show your final answer (Yes or No only) bracketed between <answer> and </answer>."
YES_NO_POSTFIX = '\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer (Yes or No only).'
YES_NO_COT_POSTFIX = " Think carefully step by step. Describe your reasoning in steps and then output yes or no at the very end."

#QA_DEFAULT_USER_PROMPT = "{question_prompt} Show your final answer bracketed between <answer> and </answer>."
QA_DEFAULT_USER_PROMPT = '{question_prompt}\nAt the very end, you **must** type "Answer:" first, then you **must** print your final answer to the question.'

letter2num = {"A": 1, "B": 2, "C": 3, "D": 4, "Z": 5}
num2letter = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}

class benchmark_base:
    def __init__(self, cot):
        self.name = "base"
        self.data_df, self.question_list, self.true_label_list = pd.DataFrame(), [], []
        self.cot = cot
    
    def save_intermediate(self, pred_label_list, model_name, column_name, eval_range=None):
        if not os.path.exists(save_intermediate_dir):
            os.makedirs(save_intermediate_dir)
        save_dir_tmp = f"{save_intermediate_dir}/{model_name}_{self.name}_results.csv"
        try:
            save_df = pd.read_csv(save_dir_tmp)
        except:
            save_df = self.data_df.copy()
        if eval_range is None:
            save_df[column_name] = pred_label_list
        else:
            save_df.loc[eval_range, column_name] = pred_label_list
        save_df.to_csv(save_dir_tmp, index=False)
    
    def clean_text(self, text):
        pattern = r"[^a-zA-Z0-9 !#$%&()*'\"+,.:;<=>?@_{|}-]"
        cleaned_text = re.sub(pattern, ' ', text)
        return re.sub(r"\s\s+", " ", cleaned_text).strip()

    def result_list_preprocessing(self, pred_text_list, result_type="multiple_choice"):
        error_num = 0
        pred_label_list = []
        for pred_text in pred_text_list:
            text = self.clean_text(pred_text)

            # Answer tag extraction
            start = text.find("<answer>") + len("<answer>") if text.find("<answer>") != -1 else 0
            end = text.find("</answer>") if text.find("</answer>") != -1 else len(text)
            text = text[start:end]
            start = text.rfind("Answer:") + len("Answer:") if text.rfind("Answer:") != -1 else -5 #Only tolerate 5 chars
            text = text[start:]
            
            if result_type == "multiple_choice":
                pattern = r'\b[A-Z]\b'
                # pattern = re.compile(r'[ABCD]')
                match = re.search(pattern, text, re.MULTILINE)
                if match:
                    pred_label_list.append(match.group(0))
                # matches = list(pattern.finditer(text))
                # if matches:
                #     if self.cot != 0:
                #         pred_label_list.append(matches[-1].group())
                #     else:
                #         pred_label_list.append(matches[0].group())
                else:
                    pred_label_list.append(text)
                    error_num += 1
            elif result_type == "yes_no":
                pattern = re.compile(r'\b(yes|no)\b', re.IGNORECASE)
                matches = list(pattern.finditer(text))
                if matches:
                    if self.cot != 0:
                        pred_label_list.append(int(matches[-1].group().lower() == "yes"))
                    else:
                        pred_label_list.append(int(matches[0].group().lower() == "yes"))
                else:
                    pred_label_list.append(text)
                    error_num += 1
            else:
                pred_label_list.append(text)

        return pred_label_list, error_num
    
    def load_question_list(self):
        return self.question_list
    
    def load_random_question_list(self, num_q=None, split="all", random_seed=None):
        train_indices, test_indices = train_test_split(list(range(len(self.question_list))), test_size=0.4, random_state=42)
        if random_seed is not None:
            random.seed(random_seed)
        if split == "all":
            if num_q is None:
                return self.question_list, None
            else:
                rand_idx = random.sample(range(len(self.question_list)), num_q)
                return [self.question_list[i] for i in rand_idx], rand_idx
        elif split == "train":
            if num_q is None:
                return [self.question_list[i] for i in train_indices], train_indices
            else:
                rand_idx = random.sample(train_indices, num_q)
                return [self.question_list[i] for i in rand_idx], rand_idx
        elif split == "test":
            if num_q is None:
                return [self.question_list[i] for i in test_indices], test_indices
            else:
                rand_idx = random.sample(test_indices, num_q)
                return [self.question_list[i] for i in rand_idx], rand_idx

    def eval_question_list(self, pred_text_list, save_intermediate=("all", "", ""), eval_range=None, return_error_idx=False):
        return dict()
    
    def get_user_prompt(self):
        if self.cot >= 1:
            return MULTIPLE_CHOICE_COT_USER_PROMPT
        else:
            return MULTIPLE_CHOICE_DEFAULT_USER_PROMPT
    
    def get_user_prompt_new(self, prompt_type="base"):
        with open(os.path.join(project_root_dir, f'./data/task_prompts/{self.name}/{prompt_type}.md'), 'r') as file:
            user_prompt = file.read()
        return user_prompt
    
    def get_max_token_len(self):
        if self.cot != 0:
            return 512
        else:
            return 16

class benchmark_arc(benchmark_base):
    def __init__(self, cot):
        self.name = "arc"
        self.data_df = pd.read_csv(os.path.join(project_root_dir, data_dir[self.name]))
        self.cot = cot

        self.question_list = []
        for idx, item in self.data_df.iterrows():
            q_text = item["question"].strip().replace("A", "(A)").replace("B", "(B)").replace("C", "(C)").replace("D", "(D)") + "\n"
            self.question_list.append(q_text)
        
        self.true_label_list = list(self.data_df["answerKey"])
        for idx in range(len(self.true_label_list)):
            self.true_label_list[idx] = self.true_label_list[idx].upper().strip()
            if self.true_label_list[idx] in num2letter:
                self.true_label_list[idx] = num2letter[self.true_label_list[idx]]
    

    def eval_question_list(self, pred_text_list, save_intermediate=("all", "", ""), eval_range=None, return_error_idx=False):
        # Save raw prediction
        if save_intermediate[0] in ["all", "raw"]: self.save_intermediate([self.clean_text(tmp_text) for tmp_text in pred_text_list], "raw_"+save_intermediate[1], save_intermediate[2], eval_range=eval_range)

        pred_label_list, error_num = self.result_list_preprocessing(pred_text_list, result_type="multiple_choice")
        
        if save_intermediate[0] in ["all", "raw"]: self.save_intermediate(pred_label_list, save_intermediate[1], save_intermediate[2], eval_range=eval_range)
        
        metrics = {}
        if save_intermediate[0] in ["all", "eval"]:
            if eval_range is None:
                local_true_label_list = self.true_label_list
            else:
                local_true_label_list = [self.true_label_list[i] for i in eval_range]
            metrics = {f"{self.name.upper()}_acc": accuracy_score(local_true_label_list, pred_label_list),}
                    # f"{self.name.upper()}_acc_no_error": (accuracy_score(local_true_label_list, pred_label_list) * len(pred_label_list)) / (len(pred_label_list) - error_num) if (len(pred_label_list) - error_num != 0) else 0,
                    # f"{self.name.upper()}_error": error_num}

            if return_error_idx:
                metrics[f"{self.name.upper()}_error_idx"] = [i for i, (a, b) in enumerate(zip(local_true_label_list, pred_label_list)) if a != b]

        return metrics

class benchmark_hellaswag(benchmark_base):
    def __init__(self, cot):
        self.name = "hellaswag"
        self.data_df = pd.read_csv(os.path.join(project_root_dir, data_dir[self.name])).sample(n=1000, random_state=42).reset_index(drop=True)
        self.cot = cot

        self.question_list = []
        for idx, item in self.data_df.iterrows():
            q_text = f"{item['ctx'].strip()}\nA. {item['endings'][0]}\nB. {item['endings'][1]}\nC. {item['endings'][2]}\nD. {item['endings'][3]}\n"
            self.question_list.append(q_text)
        
        self.true_label_list = list(self.data_df["label"])
        for idx in range(len(self.true_label_list)):
            self.true_label_list[idx] = num2letter[int(self.true_label_list[idx])+1]


    def eval_question_list(self, pred_text_list, save_intermediate=("all", "", ""), eval_range=None, return_error_idx=False):
        # Save raw prediction
        if save_intermediate[0] in ["all", "raw"]: self.save_intermediate([self.clean_text(tmp_text) for tmp_text in pred_text_list], "raw_"+save_intermediate[1], save_intermediate[2], eval_range=eval_range)

        pred_label_list, error_num = self.result_list_preprocessing(pred_text_list, result_type="multiple_choice")
        
        if save_intermediate[0] in ["all", "raw"]: self.save_intermediate(pred_label_list, save_intermediate[1], save_intermediate[2], eval_range=eval_range)
        
        metrics = {}
        if save_intermediate[0] in ["all", "eval"]:
            if eval_range is None:
                local_true_label_list = self.true_label_list
            else:
                local_true_label_list = [self.true_label_list[i] for i in eval_range]
            metrics = {f"{self.name.upper()}_acc": accuracy_score(local_true_label_list, pred_label_list),}
                    # f"{self.name.upper()}_acc_no_error": (accuracy_score(local_true_label_list, pred_label_list) * len(pred_label_list)) / (len(pred_label_list) - error_num) if (len(pred_label_list) - error_num != 0) else 0,
                    # f"{self.name.upper()}_error": error_num}
            
            if return_error_idx:
                metrics[f"{self.name.upper()}_error_idx"] = [i for i, (a, b) in enumerate(zip(local_true_label_list, pred_label_list)) if a != b]

        return metrics

class benchmark_truthfulqa(benchmark_base):
    def __init__(self, cot):
        self.name = "truthfulqa"
        self.data_df = pd.read_csv(os.path.join(project_root_dir, data_dir[self.name]))
        self.cot = cot

        self.question_list = self.data_df["question"]
        self.true_label_list = list(self.data_df["best_answer"])

        self.correct_answer_list = [lib.utils.split_multi_answer(text, add_no_comment=True) for text in self.data_df["correct_answers"]]
        self.incorrect_answer_list = [lib.utils.split_multi_answer(text) for text in self.data_df["incorrect_answers"]]

        self.bleurt = None
    
    def get_user_prompt(self):
        return QA_DEFAULT_USER_PROMPT

    def eval_question_list(self, pred_text_list, save_intermediate=("all", "", ""), eval_range=None, return_error_idx=False):
        # Save raw prediction
        if save_intermediate[0] in ["all", "raw"]: self.save_intermediate([self.clean_text(tmp_text) for tmp_text in pred_text_list], "raw_"+save_intermediate[1], save_intermediate[2], eval_range=eval_range)

        pred_label_list, _ = self.result_list_preprocessing(pred_text_list, result_type="raw")
        
        if save_intermediate[0] in ["all", "raw"]: self.save_intermediate(pred_label_list, save_intermediate[1], save_intermediate[2], eval_range=eval_range)

        metrics = {}
        if save_intermediate[0] in ["all", "eval"]:
            if eval_range is None:
                bleu_tmp = lib.utils.bleu_score(pred_label_list, self.correct_answer_list, self.incorrect_answer_list, return_error_idx)
                #rouge_tmp = lib.utils.rouge_score(pred_label_list, self.correct_answer_list, self.incorrect_answer_list)
                #if self.bleurt is None:
                #    self.bleurt = load_metric("bleurt")
                #bleurt_tmp = lib.utils.bleurt_score(pred_label_list, self.correct_answer_list, self.incorrect_answer_list, self.bleurt)
            else:
                bleu_tmp = lib.utils.bleu_score(pred_label_list, [self.correct_answer_list[i] for i in eval_range], [self.incorrect_answer_list[i] for i in eval_range], return_error_idx)
            

            metrics = {#f"{self.name.upper()}_BLEURT_acc": bleurt_tmp["BLEURT_acc"],
                    f"{self.name.upper()}_BLEU_acc": bleu_tmp["BLEU_acc"],
                    #f"{self.name.upper()}_rouge1_acc": rouge_tmp["rouge1_acc"],
                    #f"{self.name.upper()}_BLEURT_full": bleurt_tmp,
                    #f"{self.name.upper()}_BLEU_full": bleu_tmp,
                    #f"{self.name.upper()}_ROUGE_full": rouge_tmp,
                    }

            if return_error_idx:
                metrics[f"{self.name.upper()}_error_idx"] = bleu_tmp["BLEU_error_idx"]

        return metrics
    
    def get_max_token_len(self):
        return 64

class benchmark_gsm8k(benchmark_base):
    def __init__(self, cot):
        self.name = "gsm8k"
        self.data_df = pd.read_csv(os.path.join(project_root_dir, data_dir[self.name]))
        self.cot = cot

        self.question_list = list(self.data_df["question"])
        
        self.true_label_list = list(self.data_df["answer"].apply(lambda x: str(x)))

    def eval_question_list(self, pred_text_list, save_intermediate=("all", "", ""), eval_range=None, return_error_idx=False):
        # Save raw prediction
        if save_intermediate[0] in ["all", "raw"]: self.save_intermediate([self.clean_text(tmp_text) for tmp_text in pred_text_list], "raw_"+save_intermediate[1], save_intermediate[2], eval_range=eval_range)

        pred_label_list, _ = self.result_list_preprocessing(pred_text_list, result_type="raw")
        
        if save_intermediate[0] in ["all", "raw"]: self.save_intermediate(pred_label_list, save_intermediate[1], save_intermediate[2], eval_range=eval_range)
        
        metrics = {}
        if save_intermediate[0] in ["all", "eval"]:
            if eval_range is None:
                local_true_label_list = self.true_label_list
            else:
                local_true_label_list = [self.true_label_list[i] for i in eval_range]
            assert len(local_true_label_list) == len(pred_label_list)
            correct_idx = []
            for idx in range(len(local_true_label_list)):
                # If true answer start with special character
                if bool(re.match(r'^\W', local_true_label_list[idx])):
                    pattern = r'(?<!\w)' + re.escape(local_true_label_list[idx]) + r'(?!\S)'
                else:
                    pattern = r'\b' + re.escape(local_true_label_list[idx]) + r'\b'
                if re.search(pattern, pred_label_list[idx], re.IGNORECASE | re.MULTILINE):
                    correct_idx.append(idx)
            metrics = {f"{self.name.upper()}_acc": len(correct_idx)/len(local_true_label_list)}

            if return_error_idx:
                metrics[f"{self.name.upper()}_error_idx"] = [idx for idx in range(len(local_true_label_list)) if idx not in correct_idx]
        return metrics






def init_benchmark(name="mmlu", cot=0) -> benchmark_base:
    if name == "arc":
        return benchmark_arc(cot=cot)
    elif name == "hellaswag":
        return benchmark_hellaswag(cot=cot)
    elif name == "truthfulqa":
        return benchmark_truthfulqa(cot=cot)
    elif "gsm8k" in name:
        return benchmark_gsm8k(cot=cot)
