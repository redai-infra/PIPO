import re
from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig, StringExtractionConfig
from latex2sympy2_extended import NormalizationConfig


class MathEvaluator:
    def __call__(self, solution_str: str, ground_truth: str) -> bool:
        gold = parse(
            ground_truth,
            extraction_config=[ExprExtractionConfig()],
        )
        answer = parse(
            solution_str,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        boxed="last",
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                ),
                ExprExtractionConfig(),
            ],
            extraction_mode="first_match",
        )
        if len(answer) == 0:
            return False, "No extracted answer"
        else:
            return verify(gold, answer), str(answer)


class MCQEvaluator:
    # case-insensitive, must include ":"
    # only last three lines to avoid picking up earlier parts of the cot
    # re_pattern = r'`?"?[Aa]nswer"?:\s?"?([A-Za-z])"?`?'
    re_pattern = r'\\boxed{([A-Z])}'
    def __call__(self, solution_str: str, ground_truth: str) -> bool:
        pred = re.findall(self.re_pattern, '\n'.join(solution_str.strip().split('\n')[-3:]))
        # -3: incase of "\\boxed{A}\n\nOTHER-CONTENT"
        if len(pred) == 0:
            return False, "No extracted answer"
        else:
            return pred[-1] == ground_truth, pred[-1]


evaluator_map = {
    "aime2025": MathEvaluator(),
    "dapo_math": MathEvaluator(),
    "dapo_math_rl": MathEvaluator(),

    "gpqa_diamond": MCQEvaluator(),
    "lb2": MCQEvaluator(),
}
