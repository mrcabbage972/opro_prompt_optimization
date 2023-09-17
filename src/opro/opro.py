import os
from typing import List
import openai
from tqdm import tqdm

openai.api_key = os.getenv("OPENAI_API_KEY")

from gsm8k import get_dataset

from schema import ProblemExample, PromptExample
from prompts import prompt_example_template, problem_example_template, opt_prompt_template


def generate_opt_prompt(prompt_examples: List[PromptExample], problem_examples: List[ProblemExample]):
    formatted_prompt_examples = [prompt_example_template.format(prompt=prompt_example.prompt, score=prompt_example.score) 
                                 for prompt_example in prompt_examples]

    formatted_problem_examples = [problem_example_template.format(question=problem_example.question, answer=problem_example.answer) 
                                  for problem_example in problem_examples]

    formatted_opt_prompt = opt_prompt_template.format(prompt_examples='\n'.join(formatted_prompt_examples),
                                                      problem_examples='\n'.join(formatted_problem_examples))

    return formatted_opt_prompt

def generate_prompt_candidates(prompt_examples: List[PromptExample], problem_examples: List[ProblemExample]):
    opt_prompt = generate_opt_prompt(prompt_examples, problem_examples)

    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[opt_prompt],
    temperature=1.0,
    max_tokens=1024
    )


def build_problem_prompt(problem_example: ProblemExample, prompt_candidate: str):
    return f"Q: {problem_example.question}\n{prompt_candidate}\n"

def score_prompt_candidates(prompt_candidates: List[str], problem_examples: List[ProblemExample]) -> List[PromptExample]:
    for prompt_candidate in tqdm(prompt_candidates):
    
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[build_problem_prompt(p, prompt_candidate) for p in problem_examples],
        temperature=0.0,
        max_tokens=1024
        )


def seed_prompt_examples(problem_examples: List[ProblemExample]) -> List[PromptExample]:
    default_prompt_examples = ["Let's think step by step."]
    return score_prompt_candidates(default_prompt_examples, problem_examples)

def main():
    
    train_examples = get_dataset('train')
    test_examples = get_dataset('test')

    prompt_examples = seed_prompt_examples(test_examples)

    prompt_candidates = generate_prompt_candidates(prompt_examples, train_examples)
    solution_scores = score_prompt_candidates(prompt_candidates, test_examples)

if __name__ == '__main__':
    main()