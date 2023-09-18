import logging
from typing import List

import openai
from prompts import format_openai_chat_prompt
from prompts import opt_prompt_template
from prompts import problem_example_template
from prompts import prompt_example_template
from schema import ProblemExample
from schema import PromptExample

from src.opro.settings import MAX_RESPONSE_TOKENS
from src.opro.settings import MODEL_NAME

LOGGER = logging.getLogger(__name__)


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
    LOGGER.info('Generating prompt candidates')
    opt_prompt = generate_opt_prompt(prompt_examples, problem_examples)

    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=format_openai_chat_prompt(opt_prompt),
        temperature=1.0,
        max_tokens=MAX_RESPONSE_TOKENS
    )
    response_texts = [r.message['content'] for r in response.choices]
    return [r.split('<INS>')[1].split('</INS>')[0] for r in response_texts]
