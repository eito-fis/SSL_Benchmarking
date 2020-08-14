import numpy as np
import textdistance
import re
from itertools import permutations

METRIC_FUNCS = {}
def metric_func(name):
    def wrapper(func):
        METRIC_FUNCS[name] = func
        return func
    return wrapper

ASSOCIATION_FUNCS = {}
def association_func(name):
    def wrapper(func):
        ASSOCIATION_FUNCS[name] = func
        return func
    return wrapper

def calc_f1(tp, fp, fn):
    assert tp and fp, "No positive labels found!"
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if recall and precision:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0
    return precision, recall, f1

def calc_postive_negative(l, p):
    # True postive = tokens in labels and preds
    # False negative = tokens in labels but not in preds
    # False positive = tokens in preds but not in labels
    set_l, set_p = set(l), set(p)
    overlap = set_l & set_p
    only_l = set_l - overlap
    only_p = set_p - overlap
    tp, fp, fn = len(overlap), len(only_p), len(only_l)
    # _, _, f1 = calc_f1(tp, fp, fn)
    # return (tp, fp, fn), f1
    return (tp, fp, fn)

@metric_func("reconstruction")
def get_reconstruction_metrics(preds, labels, **kwargs):
    distances = []
    for pred, label in zip(preds, labels):
        distances.append(textdistance.levenshtein.normalized_distance(pred, label))
    distances = np.array(distances)
    ret_dict = {
        "Mean Norm Levenshtein Distance": np.mean(distances),
        "Median Norm Levenshtein Distance": np.median(distances),
        "Std Norm Levenshtein Deviation Distance": np.std(distances),
    }
    return ret_dict, None

@metric_func("extraction")
def get_extraction_metrics(preds, labels, **kwargs):
    delim = kwargs.get("delim", " | ")
    true_pos, false_pos, false_neg, total = 0, 0, 0, 0
    error_indicies = []
    for i, (pred, label) in enumerate(zip(preds, labels)):
        # Normalize punctuation
        label = re.sub(" ([.?!:;,])", "\\1", label)
        pred = re.sub(" ([.?!:;,])", "\\1", pred)

        split_labels = set([s.strip() for s in label.split(delim)])
        split_preds = set([s.strip() for s in pred.split(delim)])

        tp, fp, fn = calc_positive_negative(split_labels, split_preds)
        true_pos += tp
        false_neg += fp
        false_pos += fn

        if fp + fn != 0:
            error_indicies.append(i)
    precision, recall, f1 = calc_f1(true_pos, false_pos, false_neg)
    ret_dict = {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }
    return ret_dict, error_indicies

@association_func("json")
def process_json(x):
    start = 0
    i = 0
    last = None
    stack = []
    text = []
    while i < len(x):
        pos = (x.find("{", i), x.find("}", i))
        if sum(pos) == -2:
            return text
        pos = (np.inf if v == -1 else v for v in pos)
        i = min(pos)
        if x[i] == "{":
            stack.append(x[start:i].strip())
            last = "{"
        elif x[i] == "}":
            # assert last, f"Closing without open! {x}"
            if not last:
                print(f"WARNING! Closing without open! {x}")
            if last == "{":
                text.append((tuple(s for s in stack), x[start:i].strip()))
            if stack:
                stack.pop()
            else:
                print(f"WARNING! No stack found. {x}")
            last = "}"
            i += 1
        i += 1
        start = i
    return text

@metric_func("association")
def get_association_metrics(preds, labels, **kwargs):
    mode = kwargs.get("association_mode", "json")
    delim = kwargs.get("delim", "\\|")
    all_stats = []
    for i, (pred, label) in enumerate(zip(preds, labels)):
        # Normalize punctuation
        label = re.sub(" ([.?!:;,])", "\\1", label)
        pred = re.sub(" ([.?!:;,])", "\\1", pred)
        
        # text -> (tags, extracted text) tuples
        processed_labels = [ASSOCIATION_FUNCS[mode](s.strip())
                            for s in re.split(delim, label)]
        processed_preds = [ASSOCIATION_FUNCS[mode](s.strip())
                           for s in re.split(delim, pred)]

        best_f1 = 0
        best_stats = []
        # perms = list(permutations(processed_preds))
        # perms = [processed_preds]
        # cap = 1000
        # if len(perms) > cap:
        #     perms = random.choice(perms, k=cap)
        # for perm_preds in perms:
        #     stats = [compare(l, p)
        #              for l, p in zip(processed_labels, perm_preds)]
        #     avg_f1 = sum(s[1] for s in stats) / len(stats)
        #     if avg_f1 > best_f1:
        #         best_f1 = avg_f1
        #         best_stats = stats
        # # print(best_stats)
        # # print(best_f1)
        # # input()
        stats = [calc_positive_negative(l, p)
                 for l, p in zip(processed_labels, processed_preds)]
        # stats = [s[0] for s in best_stats]
        final_stats = tuple(sum(s) for s in zip(*stats))
        if final_stats:
            all_stats.append(final_stats)
    all_stats = tuple(sum(s) for s in zip(*all_stats))
    precision, recall, f1 = calc_f1(*all_stats)

    ret_dict = {
        "Token Precision": precision,
        "Token Recall": recall,
        "Token F1": f1
    }
    extraction_dict, error_indicies = get_extraction_metrics(preds, labels, delim=" | ")
    ret_dict.update(extraction_dict)
    reconstruction_dict, _ = get_reconstruction_metrics(preds, labels, **kwargs)
    ret_dict.update(reconstruction_dict)
    return ret_dict, error_indicies
