from tqdm import tqdm
import argparse
import pickle
import re


def read_file(filename, scope=-1):
    """
    Takes in a CONLL format file and returns a list of lists, wher each inner
    list is a sentence with each element a tuple of the form (word, tag)

    Scope of 0 uses the most general tag for each word, while a scope of -1
    utilizes the most specific tag of each word
    """
    assert scope == -1 or scope == 0, "scope must be -1 or 0"
    with open(filename, mode="rt") as f:
        sentences = []
        tag_counts = {}
        print("Processing...")
        for line in tqdm(f):
            stack = []
            sentence = []
            splits = line.split('\t')
            filter_fn = lambda x: True if x and x != " " else False
            splits = list(filter(filter_fn, re.split("(<[^>]*>)", line)))[:-1]
            for token in splits:
                if len(token) > 2 and token[0] == "<":
                    # Tag found
                    if token[1] == "/":
                        # Closing tag
                        tag = token[2:-1]
                        assert(len(stack) > 0), f"No tags on stack when {token} was found in {splits}"
                        assert(tag == stack[-1]), f"{token} does not match {stack[-1]} in {splits}"
                        stack.pop()
                    else:
                        # Opening tag
                        tag = token[1:-1]
                        stack.append(tag)
                elif len(token) > 0:
                    # Word found
                    # Label with all tags active in stack
                    tag = None if len(stack) == 0 else stack[scope]
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
                    data = (token, tag)
                    sentence.append(data)
            sentences.append(sentence)
    return sentences, tag_counts


def process(sentences, tag_counts, thresh=50):
    """
    Takes in sentences as generated by read_file() and converts them to a
    format that Finetune will be happy with
    """
    inputs = []
    labels = []
    for sentence in sentences:
        start = 0
        text = []
        input_dicts = []
        for word, tag in sentence:
            text.append(word)
            end = start + len(word)
            if tag and tag != "O" and tag_counts[tag] > thresh:
                input_dict = {
                    "start": start,
                    "end": end,
                    "label": tag,
                    "text": word
                }
                input_dicts.append(input_dict)
            start = end + 1
        concat_text = " ".join(text)
        if len(concat_text) < 400:
            inputs.append(concat_text)
            labels.append(input_dicts)
    return inputs, labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Train SSL')
    parser.add_argument('--filename',
                        type=str,
                        default="concat.docs")
    parser.add_argument('--name',
                        type=str,
                        default=None)
    parser.add_argument('--scope',
                        type=int,
                        default=-1)
    args = parser.parse_args()
    filename = args.filename

    sentences, tag_counts = read_file(filename, scope=args.scope)
    inputs, labels = process(sentences, tag_counts)
    final_count = {}
    for s in labels:
        for i in s:
            tag = i["label"]
            final_count[tag] = final_count.get(tag, 0) + 1

    save_dict = {
        "inputs": inputs,
        "labels": labels
    }
    name = "processe" if args.name is None else f"processed_{args.name}.pickle"
    with open(name, mode="wb") as f:
        pickle.dump(save_dict, f)