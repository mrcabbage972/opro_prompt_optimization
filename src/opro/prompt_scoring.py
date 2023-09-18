import logging
import re
from multiprocessing.pool import ThreadPool
from typing import List

import openai
from prompts import build_problem_prompt
from prompts import format_openai_chat_prompt
from schema import ProblemExample
from schema import PromptExample
from tqdm import tqdm

from src.opro.settings import FINAL_ANSWER_SEP
from src.opro.settings import MAX_RESPONSE_TOKENS
from src.opro.settings import MODEL_NAME
from src.opro.settings import THREADS

LOGGER = logging.getLogger(__name__)


def generate_answer_proc(candidate_problem_prompt: str):
    try:
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=format_openai_chat_prompt(candidate_problem_prompt),
            temperature=0.0,
            max_tokens=MAX_RESPONSE_TOKENS
        )
        response_text = response.choices[0].message.content
        answer = response_text.split(FINAL_ANSWER_SEP)[-1]
        answer = re.findall(r'[\d]+[.,\d]+|[\d]*[.][\d]+|[\d]+', answer)[0].replace(',', '')
        return answer
    except Exception as e:
        LOGGER.error(f'Error generating answer: {e}')
        return None


def generate_answers(demo_examples, test_examples, prompt_candidate):
    LOGGER.info(f'Generating answers for prompt candidate: {prompt_candidate}')
    pool = ThreadPool(processes=THREADS)
    candidate_problem_prompts = [(build_problem_prompt(p, demo_examples, prompt_candidate)) for p in test_examples]
    answers = []

    async_results = pool.map(generate_answer_proc, candidate_problem_prompts)
    for answer in async_results:
        assert answer is not None
        answers.append(answer)

    return answers


def get_prompt_candidate_score(generated_answers, problem_examples):
    assert len(generated_answers) == len(problem_examples)
    ground_truth_answers = [p.answer.split(FINAL_ANSWER_SEP)[-1].strip().replace(',', '') for p in problem_examples]
    return sum([1 if a == b else 0 for a, b in zip(generated_answers, ground_truth_answers)]) \
        / len(ground_truth_answers)


def score_prompt_candidates(prompt_candidates: List[str],
                            demo_examples: List[ProblemExample],
                            test_examples: List[ProblemExample]) \
        -> List[PromptExample]:
    scored_prompt_candidates = []
    for prompt_candidate in tqdm(prompt_candidates, desc='Scoring prompt candidates'):
        answers = generate_answers(demo_examples, test_examples, prompt_candidate)
        prompt_candidate_score = get_prompt_candidate_score(answers, test_examples)
        scored_prompt_candidates.append(PromptExample(prompt=prompt_candidate, score=prompt_candidate_score))
    return scored_prompt_candidates
