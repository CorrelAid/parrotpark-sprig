# from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
# from vllm import LLM, SamplingParams
# import torch
from tqdm import tqdm
import openai
import dspy
import asyncio
from typing import List, Union
from tqdm.asyncio import tqdm_asyncio

llama2_template = (
    """<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n{user_prompt} [/INST]"""
)
llama3_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""
mixtral_template = """<s>[INST]{system_prompt}\n\n{user_prompt}[/INST]"""
dbrx_template = """<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"""
jamba_template = """<|startoftext|>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"""
qwen_template = """<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"""
gemma_template = """<bos><start_of_turn>user\n{system_prompt}\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n"""
commandR_template = """<BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{system_prompt}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{user_prompt}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"""

llm_template_dict = {
    "llama-2": llama2_template,
    "llama-3": llama3_template,
    "mistral": mixtral_template,
    "mixtral": mixtral_template,
    "dbrx": dbrx_template,
    "jamba": jamba_template,
    "qwen": qwen_template,
    "gemma": gemma_template,
    "command-r": commandR_template,
}

def validate_answer(example, pred, trace=None):
    return example["extracted_apps"] == pred["extracted_apps"]

class inference_model:
    def __init__(self, base_url, model_name, api_key):
        self.base_url = base_url
        self.model_name = model_name

        self.model_type = "mistral"

        for key in llm_template_dict:
            if key in self.model_name.lower():
                self.model_type = key

        # Initialize both sync and async clients
        self.client = openai.OpenAI(api_key=api_key, base_url=self.base_url)
        self.async_client = openai.AsyncOpenAI(api_key=api_key, base_url=self.base_url)

    async def generate(
    self, 
    answer_prompts: Union[str, List[str]], 
    max_token_len: int = 3000,
    temperature: float = 0.0,
    max_concurrent=30,
    show_progress: bool = True
) -> List[str]:
        prompts = [answer_prompts] if isinstance(answer_prompts, str) else answer_prompts

        sem = asyncio.Semaphore(max_concurrent) if max_concurrent else None

        async def generate_single(prompt, total_prompts, current_idx):
            """Generate a single prompt with semaphore"""
            try:
                if sem:
                    async with sem:
                        response = await self.async_client.completions.create(
                            model=self.model_name,
                            prompt=prompt,
                            max_tokens=max_token_len,
                            temperature=temperature
                        )
                else:
                    response = await self.async_client.completions.create(
                        model=self.model_name,
                        prompt=prompt,
                        max_tokens=max_token_len,
                        temperature=temperature
                    )

                # Print progress every 50 generations
                if show_progress and current_idx % 50 == 0:
                    print(f"Progress: {current_idx}/{total_prompts} generations completed")

                return response.choices[0].text.strip()

            except Exception as e:
                print(f"Generation error for prompt: {e}")
                raise

        try:
            # Create tasks with index tracking
            tasks = [generate_single(prompt, len(prompts), idx) for idx, prompt in enumerate(prompts, 1)]
            results = await asyncio.gather(*tasks)

            # Print final progress
            if show_progress:
                print(f"Completed all {len(prompts)} generations")

            return results

        except Exception as e:
            print(f"Batch generation error: {e}")
            return []


    def get_prompt_template(self):
        return llm_template_dict[self.model_type]

class para_model:
    def __init__(self, base_url, model_name, api_key):
        self.base_url = base_url
        self.model_name = model_name

        self.model_type = "llama"

        for key in llm_template_dict:
            if key in self.model_name.lower():
                self.model_type = key

        lm = dspy.LM(
            f"openai/{self.model_name}",
            cache=True,
            api_key=api_key,
            base_url=self.base_url,
        )

        dspy.configure(lm=lm)

        self.client = lm

        class Rephraser(dspy.Signature):
            """Rephrase a given text, deviating from already explored alternatives optionally given as input."""
            text: str = dspy.InputField()
            explored_alternatives: List[str] = dspy.InputField()
            paraphrased_text: str = dspy.OutputField()

        examples = [
            dspy.Example(
                text="The ultimate test of your knowledge is your capacity to convey it to another.",
                explored_alternatives=[
                    "The test of your knowledge is your ability to convey it.",
                    "The ability to convey your knowledge is the ultimate test of your knowledge.",
                    "The ability to convey your knowledge is the most important test of your knowledge.",
                    "The test of your knowledge is how well you can convey it.",
                ],
                paraphrased_text="Your capacity to convey your knowledge is the ultimate test of it.",
            ).with_inputs("text"),
            dspy.Example(
                text="Success is not final, failure is not fatal: it is the courage to continue that counts.",
                explored_alternatives=None,
                paraphrased_text="It is the courage to continue that counts, as success is not final and failure is not fatal.",
            ).with_inputs("text"),
            dspy.Example(
                text="In three words I can sum up everything I've learned about life: it goes on.",
                explored_alternatives=[
                    "Everything I have learned about life can be summed up in three words: it goes on.",
                    "I can summarize all my life lessons in three words: it goes on.",
                    "All my life learnings can be condensed to three words: it goes on.",
                    "Three words summarize everything learned about life: it goes on.",
                ],
                paraphrased_text="Everything I have learned about life can be summed up in three words: it goes on.",
            ).with_inputs("text"),
        ]
        prog = dspy.Predict(Rephraser)
        optimizer = dspy.LabeledFewShot()

        self.program = optimizer.compile(student=prog, trainset=examples, sample=False)

    def rephrase(self,input_text, num_return_sequences):
        paras = []
        for i in range(num_return_sequences):
            temp = self.program(text=input_text,explored_alternatives=paras).paraphrased_text
            paras.append(temp)
        return paras