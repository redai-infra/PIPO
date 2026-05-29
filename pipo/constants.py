MODEL_PATH_DICT = {
    'Qwen3p5-9B': 'Qwen/Qwen3.5-9B',
    'Qwen3p5-4B': 'Qwen/Qwen3.5-4B',
    'Qwen3p5-0p8B': 'Qwen/Qwen3.5-0.8B',
}


class PROMPT:
    # prompt templates
    MATH_QUERY_TEMPLATE = """
Please reason step by step, and put your final answer within \\boxed{{}}.

{Question}
""".strip()

    MCQ_QUERY_TEMPLATE = """
Please show your final answer within \\boxed{{}} with only the choice letter, e.g., \\boxed{{A}}.

{Question}
""".strip()
#     MCQ_QUERY_TEMPLATE = """
# Please show your choice in the `answer` field with only the choice letter in the last line of your response, e.g.,
# `"answer": "C"`.

# {Question}
# """.strip()
# Qwen3.5 may NOT stably follow the instruction after long-term reasoning.

    @staticmethod
    def get_lcb_prompt(question_content, starter_code):
        prompt = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"
        prompt += f"Question: {question_content}\n\n"
        if starter_code:
            prompt += f"You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n"
            prompt += f"```python\n{starter_code}\n```\n\n"
        else:
            prompt += f"Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
            prompt += f"```python\n# YOUR CODE HERE\n```\n\n"
        return prompt
