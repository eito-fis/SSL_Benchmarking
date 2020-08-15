import numpy as np
import textdistance
import re
from itertools import permutations

PROCESS_RULES = {
    "original": [(r"(<) ", r"\1")],
    "json": [],
    "newline": [(r"({) ", r"\1"), (r"} }", "}}"), (r"{", "\n"), (r"}", "\t"),
                (r" \| ", "\n|\n")],
    None: [],
}

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
    if tp + fp == 0:
        print("No positive labels predicted!")
        return 0, 0, 0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if recall and precision:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0
    return precision, recall, f1

def calc_positive_negative(l, p):
    # True positive = tokens in labels and preds
    # False negative = tokens in labels but not in preds
    # False positive = tokens in preds but not in labels
    set_l, set_p = set(l), set(p)
    overlap = set_l & set_p
    only_l = set_l - overlap
    only_p = set_p - overlap
    tp, fp, fn = len(overlap), len(only_p), len(only_l)
    return tp, fp, fn

def check_level(splits, i):
    j = i + 1
    while j < len(splits):
        if len(splits[j]) > 2 and splits[j][0] == "<":
            if splits[j][1] == "/":
                return False
            else:
                return True
        j += 1
    return True

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

        split_labels = [s.strip() for s in label.split(delim)]
        split_preds = [s.strip() for s in pred.split(delim)]

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

@metric_func("association")
def get_association_metrics(preds, labels, **kwargs):
    mode = kwargs.get("association_mode", "json")
    delim = kwargs.get("delim", r"\|")

    all_stats = []
    for i, (pred, label) in enumerate(zip(preds, labels)):
        # Normalize punctuation
        label = re.sub(" ([.?!:;,])", r"\1", label)
        pred = re.sub(" ([.?!:;,])", r"\1", pred)
        
        # text -> ((tags,), extracted text) tuples
        processed_labels = [ASSOCIATION_FUNCS[mode](s.strip())
                            for s in re.split(delim, label)]
        processed_preds = [ASSOCIATION_FUNCS[mode](s.strip())
                           for s in re.split(delim, pred)]

        stats = [calc_positive_negative(l, p)
                 for l, p in zip(processed_labels, processed_preds)]
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

@association_func("original")
def process_original(x):
    closed = False
    text = []
    stack = []
    filter_fn = lambda x: True if x and x != " " else False
    splits = list(filter(filter_fn, re.split("(<[^>]*>)", x)))
    for i in range(len(splits)):
        token = splits[i]
        if len(token) > 2 and token[0] == "<":
            if token[1] == "/":
                # Closing tag
                closed = True
                tag = token[2:-1]
                if len(stack) == 0:
                    print(f"Warning! No tags on stack when closing!")
                elif tag != stack[-1]:
                    print(f"Warning! Tag does not match stack!")
                else:
                    stack.pop()
            else:
                # Opening tag
                closed = False
                tag = token[1:-1]
                stack.append(tag)
                high_level = check_level(splits, i)
        elif len(token) > 0:
            # Word found
            token = re.sub(" ([.?!:;,])", "\\1", token)
            if len(stack) == 0 or closed or high_level:
                    continue
            text.append((tuple(s for s in stack), token.strip()))
    return text

@association_func("newline")
def process_newline(x):
    closed = False
    details = []
    stack = []
    i = 0
    while i != -1 and i < len(x):
        end = x.find("\n", i + 1)
        if end == -1:
            span = x[i:]
        else:
            span = x[i:end]

        text_start = re.search(r"[^\n\t]", span)
        if not text_start:
            print("Warning! New detail started at end of prediction")
        else:
            text_start = text_start.start()

        prefix = span[:text_start]
        num_tab = len(re.findall("\t", prefix))
        stack = stack[:num_tab]

        text = span[text_start:].split(" < ")
        tag = text[0].strip()
        stack.append(tag)
        if len(text) == 2:
            details.append((tuple(s for s in stack), text[1].strip()))
        i = end
    return details
