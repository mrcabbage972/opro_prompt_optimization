import os
from typing import List

import openai
from dotenv import load_dotenv
from gsm8k import get_dataset
from prompts import opt_prompt_template
from prompts import problem_example_template
from prompts import prompt_example_template
from schema import ProblemExample
from schema import PromptExample
from tqdm import tqdm

# TODO: move to config class
MAX_EXAMPLES = 10
MAX_ITER = 5
MAX_RESPONSE_TOKENS = 128


def generate_opt_prompt(prompt_examples: List[PromptExample], problem_examples: List[ProblemExample]):
    formatted_prompt_examples = [
        prompt_example_template.format(prompt=prompt_example.prompt, score=prompt_example.score)
        for prompt_example in prompt_examples]

    formatted_problem_examples = [
        problem_example_template.format(question=problem_example.question, answer=problem_example.answer)
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
        max_tokens=MAX_RESPONSE_TOKENS
    )
    return response.choices


def build_problem_prompt(problem_example: ProblemExample, prompt_candidate: str):
    return f"Q: {problem_example.question}\nA: {prompt_candidate}\n"


def score_prompt_candidates(prompt_candidates: List[str], problem_examples: List[ProblemExample]) \
        -> List[PromptExample]:
    for prompt_candidate in tqdm(prompt_candidates):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[build_problem_prompt(p, prompt_candidate) for p in problem_examples],
            temperature=0.0,
            max_tokens=MAX_RESPONSE_TOKENS
        )
    return response.choices


def seed_prompt_examples(problem_examples: List[ProblemExample]) -> List[PromptExample]:
    default_prompt_examples = ["Let's think step by step."]
    return score_prompt_candidates(default_prompt_examples, problem_examples)


def update_prompt_examples(prompt_examples, scored_prompt_candidates):
    pass


def main():
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    train_examples = get_dataset('train')[:MAX_EXAMPLES]
    test_examples = get_dataset('test')[:MAX_EXAMPLES]

    prompt_examples = seed_prompt_examples(test_examples)

    for iter in range(MAX_ITER):
        prompt_candidates = generate_prompt_candidates(prompt_examples, train_examples)
        scored_prompt_candidates = score_prompt_candidates(prompt_candidates, test_examples)
        prompt_examples = update_prompt_examples(prompt_examples, scored_prompt_candidates)


if __name__ == '__main__':
    main()
