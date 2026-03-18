"""
Math answer extraction and comparison utilities.
Supports two scoring modes:
  - exact_match: normalized string / numeric comparison
  - math_verify: symbolic equivalence via HuggingFace math-verify (SymPy-based)
"""
import re
import string


# ===================== Answer Extraction =====================

def extract_boxed_answer(text):
    """Extract the last \\boxed{...} content from text, handling nested braces."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        idx = text.rfind("\\boxed ")
        if idx == -1:
            return None
        rest = text[idx + len("\\boxed "):]
        return rest.split()[0] if rest.split() else None

    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth == 0:
        return text[start:i - 1]
    return None


def extract_last_number(text):
    """Extract the last number from text as a fallback."""
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        return numbers[-1]
    return None


def extract_answer_from_response(response):
    """
    Extract the final answer from a model response.
    Tries \\boxed{} first, then falls back to other patterns.
    """
    boxed = extract_boxed_answer(response)
    if boxed is not None:
        return boxed

    patterns = [
        r"[Tt]he\s+(?:final\s+)?answer\s+is\s*:?\s*\$?([^$\n.]+)\$?",
        r"[Aa]nswer\s*[:=]\s*\$?([^$\n.]+)\$?",
        r"[Tt]herefore,?\s*\$?([^$\n.]+)\$?",
        r"[Hh]ence,?\s*\$?([^$\n.]+)\$?",
    ]
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            return match.group(1).strip()

    return extract_last_number(response)


# ===================== Exact Match =====================

def normalize_math_answer(answer):
    """Normalize a math answer string for comparison."""
    if answer is None:
        return ""
    answer = str(answer).strip()

    answer = answer.strip("$")

    answer = re.sub(r"\\text\{([^}]*)\}", r"\1", answer)
    answer = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", answer)
    answer = re.sub(r"\\textbf\{([^}]*)\}", r"\1", answer)

    answer = re.sub(r"\\(?:d|t)?frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", answer)

    answer = answer.replace("\\left", "").replace("\\right", "")
    answer = answer.replace("\\,", "").replace("\\;", "").replace("\\!", "")
    answer = answer.replace("\\quad", " ").replace("\\qquad", " ")

    answer = answer.replace("\\infty", "infinity")
    answer = answer.replace("\\pi", "pi")
    answer = answer.replace("\\cdot", "*")
    answer = answer.replace("\\times", "*")
    answer = answer.replace("\\div", "/")
    answer = answer.replace("\\pm", "+-")
    answer = answer.replace("\\mp", "-+")
    answer = answer.replace("\\%", "%")
    answer = answer.replace("\\$", "$")
    answer = answer.replace("\\le", "<=")
    answer = answer.replace("\\ge", ">=")
    answer = answer.replace("\\neq", "!=")
    answer = answer.replace("\\ne", "!=")
    answer = answer.replace("\\approx", "≈")

    answer = re.sub(r"\{([^{}]*)\}", r"\1", answer)

    answer = " ".join(answer.split())
    answer = answer.lower()
    answer = answer.rstrip(".")

    return answer.strip()


def try_parse_number(s):
    """Try to parse a string as a number. Returns (success, value)."""
    s = s.strip().replace(",", "")
    try:
        return True, float(s)
    except ValueError:
        return False, None


def math_equal(pred_answer, gt_answer):
    """
    Check if predicted answer equals ground truth answer.
    Handles numeric comparison, LaTeX normalization, and string matching.
    """
    if pred_answer is None or gt_answer is None:
        return False

    pred_norm = normalize_math_answer(pred_answer)
    gt_norm = normalize_math_answer(gt_answer)

    if pred_norm == gt_norm:
        return True

    pred_ok, pred_val = try_parse_number(pred_norm)
    gt_ok, gt_val = try_parse_number(gt_norm)
    if pred_ok and gt_ok:
        if abs(pred_val - gt_val) < 1e-6:
            return True
        if gt_val != 0 and abs((pred_val - gt_val) / gt_val) < 1e-6:
            return True

    try:
        pred_eval = eval(pred_norm.replace("^", "**"))
        gt_eval = eval(gt_norm.replace("^", "**"))
        if isinstance(pred_eval, (int, float)) and isinstance(gt_eval, (int, float)):
            if abs(pred_eval - gt_eval) < 1e-6:
                return True
    except:
        pass

    return False


def exact_match_score(prediction, ground_truth, **kwargs):
    """
    Exact match scoring: extract answer from response, normalize, compare.
    Returns 1.0 if correct, 0.0 if incorrect.
    """
    pred_answer = extract_answer_from_response(prediction)
    return 1.0 if math_equal(pred_answer, ground_truth) else 0.0


# ===================== math_verify (SymPy-based) =====================

def math_verify_score(prediction, ground_truth, **kwargs):
    """
    Symbolic equivalence scoring via math-verify library.
    Uses SymPy-based parsing to check mathematical equivalence.
    Returns 1.0 if correct, 0.0 if incorrect.
    """
    from math_verify import parse, verify

    try:
        gold_parsed = parse(ground_truth)
    except Exception:
        # If ground truth can't be parsed, fall back to exact match
        return exact_match_score(prediction, ground_truth, **kwargs)

    try:
        pred_parsed = parse(prediction)
    except Exception:
        return 0.0

    try:
        return 1.0 if verify(gold_parsed, pred_parsed) else 0.0
    except Exception:
        return 0.0
