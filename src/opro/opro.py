from typing import List

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

def main():
    prompt_examples = []
    problem_examples = []

    opt_prompt = generate_opt_prompt(prompt_examples, problem_examples)

if __name__ == '__main__':
    main()