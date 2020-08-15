from tqdm import tqdm
import argparse
import pickle
import re

def check_level(splits, i):
    j = i + 1
    high_level = True
    while j < len(splits):
        if len(splits[j]) > 2 and splits[j][0] == "<":
            if splits[j][1] == "/":
                high_level = False
            break
        j += 1
    return high_level

def process_json(line):
    closed = False
    entities = []
    entity = ""
    stack = []
    sentence = []
    filter_fn = lambda x: True if x and x != " " else False
    splits = list(filter(filter_fn, re.split("(<[^>]*>)", line)))[:-1]
    for i in range(len(splits)):
        token = splits[i]
        if len(token) > 2 and token[0] == "<":
            # Tag found
            if token[1] == "/":
                # Closing tag
                closed = True
                tag = token[2:-1]
                assert(len(stack) > 0), f"No tags on stack when {token} was found in {splits}"
                assert(tag == stack[-1]), f"{token} does not match {stack[-1]} in {splits}"
                stack.pop()
                entity += "}, "
                if len(stack) == 0:
                    if len(entity):
                        entities.append(entity)
                        entity = ""
            else:
                # Opening tag
                closed = False
                tag = token[1:-1]
                stack.append(tag)
                entity = entity + (" " if len(entity) and entity[-1] != " " else "") + tag + " {"
                high_level = check_level(splits, i)
        elif len(token) > 0:
            # Word found
            # Label with all tags active in stack
            token = re.sub(" ([.?!:;,])", "\\1", token)
            sentence.append(token)
            if len(entity) == 0 or len(stack) == 0 or closed or high_level:
                continue
            entity += token
    return "| ".join(entities), " ".join(sentence)

def process_original(line):
    entities = []
    entity = ""
    stack = []
    sentence = []
    filter_fn = lambda x: True if x and x != " " else False
    splits = list(filter(filter_fn, re.split("(<[^>]*>)", line)))[:-1]
    closed = False
    new_open = True
    bottom_level = True
    for i in range(len(splits)):
        token = splits[i]
        if len(token) > 2 and token[0] == "<":
            # Tag found
            if token[1] == "/":
                # Closing tag
                tag = token[2:-1]
                assert(len(stack) > 0), f"No tags on stack when {token} was found in {splits}"
                assert(tag == stack[-1]), f"{token} does not match {stack[-1]} in {splits}"
                stack.pop()
                closed = True
                entity += " </" + tag + ">"
                if len(stack) == 0:
                    if len(entity):
                        entities.append(entity)
                        entity = ""
            else:
                # Opening tag
                tag = token[1:-1]
                stack.append(tag)
                entity = entity + " <" + tag + ">"
                closed = False
                new_open = True
                high_level = check_level(splits, i)
        elif len(token) > 0:
            token = re.sub(" ([.?!:;,])", "\\1", token).strip()
            sentence.append(token)
            if len(entity) == 0 or len(stack) == 0 or closed or high_level:
                continue
            entity += " " + token
            new_open = False
    return " |".join(entities).strip(), " ".join(sentence).strip()

def process_newline(line):
    # { = \n, } = \t
    closed = False
    entities = []
    entity = ""
    stack = []
    sentence = []
    filter_fn = lambda x: True if x and x != " " else False
    splits = list(filter(filter_fn, re.split("(<[^>]*>)", line)))[:-1]
    for i in range(len(splits)):
        token = splits[i]
        if len(token) > 2 and token[0] == "<":
            # Tag found
            if token[1] == "/":
                # Closing tag
                closed = True
                tag = token[2:-1]
                assert(len(stack) > 0), f"No tags on stack when {token} was found in {splits}"
                assert(tag == stack[-1]), f"{token} does not match {stack[-1]} in {splits}"
                stack.pop()
                if len(stack) == 0:
                    if len(entity):
                        entities.append(entity)
                        entity = ""
            else:
                # Opening tag
                closed = False
                tag = token[1:-1]
                prefix = " {" + ("}" * len(stack)) + " "
                entity = entity + (prefix if len(entity) else "") + tag
                stack.append(tag)
                high_level = check_level(splits, i)
                if not high_level:
                    entity += " < "
        elif len(token) > 0:
            # Word found
            # Label with all tags active in stack
            token = re.sub(" ([.?!:;,])", "\\1", token).strip()
            sentence.append(token)
            if len(entity) == 0 or len(stack) == 0 or closed or high_level:
                continue
            entity += token
    return " | ".join(entities), " ".join(sentence)

def read_file(filename, mode="equals", debug=False):
    with open(filename, mode="rt") as f:
        mode2func = {
            "json": process_json,
            "original": process_original,
            "newline": process_newline,
        }
        all_entities = []
        inputs = []
        print("Processing...")
        for line in tqdm(f):
            assert mode in mode2func, "Not a processing mode!"
            if debug:
                print(line)
                input()
            entities, sentence = mode2func[mode](line)
            all_entities.append(entities)
            inputs.append(sentence)

            # if len(entities) > 512:
            #     print("Long entity found!")
            #     # print(entities)
            # if len(sentence) > 512:
            #     print("Long sentence found!")
            #     # print(sentence)
            if debug:
                print(entities)
                print(sentence)
                input()
    return inputs, all_entities

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Train SSL')
    parser.add_argument('--filename',
                        type=str,
                        default="concat.docs")
    parser.add_argument('--name',
                        type=str,
                        default="processed_t5.pickle")
    parser.add_argument('--mode',
                        type=str,
                        default="equals")
    parser.add_argument('--debug',
                        action="store_true",
                        default=False)
    args = parser.parse_args()
    filename = args.filename

    inputs, labels = read_file(filename, mode=args.mode, debug=args.debug)

    _in = None
    for i, l in zip(inputs, labels):
        print(f"Input: {i}")
        # _l = l.replace("{", "\n").replace("}", "\t")
        # print(f"Label:\n{_l}")
        print(f"Label:\n{l}")
        print("=" * 20)
        if not _in:
            _in = input()

    save_dict = {
        "inputs": inputs,
        "labels": labels
    }
    with open(args.name, mode="wb") as f:
        pickle.dump(save_dict, f)
