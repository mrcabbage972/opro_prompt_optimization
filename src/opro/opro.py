from typing import List
from pydantic import BaseModel

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

class PromptExample(BaseModel):
    prompt: str
    score: float
    

class ProblemExample(BaseModel):
    question: str
    answer: str



def generate_opt_prompt(prompt_examples: List[PromptExample], problem_examples: List[ProblemExample]):
    formatted_prompt_examples = [prompt_example_template.format(prompt=prompt_example.prompt, score=prompt_example.score) 
                                 for prompt_example in prompt_examples]

    formatted_problem_examples = [problem_example_template.format(question=problem_example.question, answer=problem_example.answer) 
                                  for problem_example in problem_examples]

    formatted_opt_prompt = opt_prompt_template.format(prompt_examples='\n'.join(formatted_prompt_examples),
                                                      problem_examples='\n'.join(formatted_problem_examples))

    return formatted_opt_prompt

def main():
    prompt_examples = []
    problem_examples = []

    opt_prompt = generate_opt_prompt(prompt_examples, problem_examples)

if __name__ == '__main__':
    main()