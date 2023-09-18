from typing import List

import openai
from prompts import format_openai_chat_prompt
from prompts import opt_prompt_template
from prompts import problem_example_template
from prompts import prompt_example_template
from schema import ProblemExample
from schema import PromptExample

from src.opro.settings import MAX_RESPONSE_TOKENS


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
        messages=[format_openai_chat_prompt(opt_prompt)],
        temperature=1.0,
        max_tokens=MAX_RESPONSE_TOKENS
    )
    return response.choices
