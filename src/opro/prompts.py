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