"""
RULER evaluation metrics.
Ported from ShadowKV's data/metrics.py.
"""

import re
import string


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def postprocess_pred(predict_str):
    predict_str = predict_str.strip().replace('<|eot_id|>', '').replace('</s>', '').replace('</s', '').replace('</', '')
    np_pattern = re.compile(r'[\x00-\x1f]')
    predict_str = np_pattern.sub('\n', predict_str).strip()
    return predict_str


def needle_score(prediction, ground_truth):
    prediction = normalize_answer(postprocess_pred(prediction))
    ground_truth = normalize_answer(ground_truth)
    min_length = len(ground_truth)
    score = float(prediction[:min_length] == ground_truth[:min_length])
    pred_list = prediction.split()
    score = max(float(ground_truth in pred_list), score)
    return score


def multi_number(prediction, ground_truth):
    prediction = normalize_answer(prediction)
    prediction_list = re.findall(r'\d+', prediction)
    hits = [item for item in ground_truth if item in prediction_list]
    return len(hits) / len(ground_truth) if ground_truth else 0.0


def multi_words(prediction, ground_truth):
    prediction = prediction.lower()
    ground_truth = [gt.lower() for gt in ground_truth]
    prediction_list = re.findall(r'\b\w+\b', prediction)
    hits = [item for item in ground_truth if item in prediction_list]
    return len(hits) / len(ground_truth) if ground_truth else 0.0


def string_match_part(prediction, ground_truth):
    prediction = postprocess_pred(prediction)
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]
    score_ref_in_pred = max(1.0 if r.lower() in prediction.lower() else 0.0 for r in ground_truth)
    score_pred_in_ref = max(1.0 if prediction.lower() in r.lower() else 0.0 for r in ground_truth)
    return round(max(score_ref_in_pred, score_pred_in_ref), 2)
