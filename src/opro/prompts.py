from typing import List

from src.opro.schema import ProblemExample

opt_prompt_template = """
Your task is to generate the instruction <INS>. Below are some previous instructions with their scores.
The score ranges from 0 to 100.
{prompt_examples}
Below are some problems.
Problem:
{problem_examples}
Generate an instruction that is different from all the instructions <INS> above, and has a higher score
than all the instructions <INS> above. The instruction should begin with <INS> and end with </INS>.
The instruction should be concise, effective, and generally applicable to all problems above
"""

prompt_example_template = """
text:
{prompt}
score:
{score}

"""

problem_example_template = """
Q: {question}
A: <INS>
Ground truth answer:
{answer}
"""

qna_prompt_format = """
Q: {question}
A: {instruction}
{answer}
"""


def build_problem_prompt(new_problem: ProblemExample, demo_examples: List[ProblemExample], prompt_candidate: str):
    formatted_prompt_examples = '\n'.join([qna_prompt_format.format(question=p.question,
                                                                    instruction=prompt_candidate,
                                                                    answer=p.answer)
                                           for p in demo_examples])
    new_problem = qna_prompt_format.format(question=new_problem.question,
                                           instruction=prompt_candidate,
                                           answer="")
    meta_instruction = "Solve the problem in the same format as in the following examples."
    return f"{meta_instruction}\nExamples:\n {formatted_prompt_examples}\n Problem:\n {new_problem}"


def format_openai_chat_prompt(prompt_text: str):
    return [{'role': 'user', 'content': prompt_text}]
