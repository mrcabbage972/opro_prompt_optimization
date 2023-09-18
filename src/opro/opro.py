import os
from typing import List

import openai
from dotenv import load_dotenv
from gsm8k import get_dataset
from schema import ProblemExample
from schema import PromptExample

from src.opro.prompt_generation import generate_prompt_candidates
from src.opro.prompt_scoring import score_prompt_candidates
from src.opro.settings import MAX_EXAMPLES
from src.opro.settings import MAX_ITER


def seed_prompt_examples(demo_examples: List[ProblemExample], test_examples: List[ProblemExample])\
        -> List[PromptExample]:
    default_prompt_examples = ["Let’s solve the problem."]
    return score_prompt_candidates(default_prompt_examples, demo_examples, test_examples)


def update_prompt_examples(previous_prompt_examples: List[PromptExample],
                           scored_prompt_candidates: List[PromptExample]) -> List[PromptExample]:
    pass


def main():
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    train_examples = get_dataset('train')[:MAX_EXAMPLES]
    test_examples = get_dataset('test')[:MAX_EXAMPLES]

    prompt_examples = seed_prompt_examples(train_examples[:1], test_examples)

    for iteration in range(MAX_ITER):
        prompt_candidates = generate_prompt_candidates(prompt_examples, train_examples)
        scored_prompt_candidates = score_prompt_candidates(prompt_candidates, train_examples, test_examples)
        prompt_examples = update_prompt_examples(prompt_examples, scored_prompt_candidates)


if __name__ == '__main__':
    main()
