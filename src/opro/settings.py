# Quote from paper on hyperparams
# ===============================
# Implementation details. We set the temperature to be 0 when evaluating the performance of
# generated instructions, in which case the scorer LLM greedily decodes. Unless otherwise specified, we
# set the default temperature to be 1.0 for optimizer LLMs to generate diverse and creative instructions.
# At each optimization step, we prompt the optimizer LLM with the meta-prompt 8 times to generate 8
# instructions, then we add these instructions with their training scores to the optimization trajectory
# in the meta-prompt. Our meta-prompt at each step contains the best 20 instructions so far and 3
# randomly picked exemplars from the training set. We study the effect of different hyperparameters in
# ablation studies (Section 5.3). Appendix C.2 presents the full meta-prompts for different optimizer
# LLMs.
# TODO: move to config class
MODEL_NAME = "gpt-3.5-turbo"
MAX_TRAIN_EXAMPLES = 5
MAX_TEST_EXAMPLES = 15
MAX_ITER = 5
MAX_RESPONSE_TOKENS = 1024
MAX_PROMPT_CANDIDATES = 20
CANDIDATES_PER_STEP = 8
THREADS = 1

FINAL_ANSWER_SEP = 'Final Answer: '
