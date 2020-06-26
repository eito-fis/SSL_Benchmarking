from tqdm import tqdm
import pickle


def read_file(filename):
    """
    Takes in a CONLL format file and returns a list of lists, wher each inner
    list is a sentence with each element a tuple of the form (word, tag)
    """
    with open(filename, mode="rt") as f:
        sentence = []
        sentences = []
        print("Processing...")
        for line in tqdm(f):
            if (len(line) == 0 or line.startswith('-DOCSTART') or
                    line[0] == "\n"):
                if len(sentence) > 0:
                    sentences.append(sentence)
                    sentence = []
                continue
            splits = line.split(' ')
            data = [splits[0], splits[-1][:-1]]
            # if data[1] == "O":
            #     data[1] = "<PAD>"
            sentence.append(tuple(data))
    return sentences


def process(sentences):
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
            if tag != "O":
                input_dict = {
                    "start": start,
                    "end": end,
                    "label": tag,
                    "text": word
                }
                input_dicts.append(input_dict)
            start = end + 1
        inputs.append(" ".join(text))
        labels.append(input_dicts)
    return inputs, labels

if __name__ == "__main__":
    sentences = read_file("concat.txt")
    inputs, labels = process(sentences)
    save_dict = {
        "inputs": inputs,
        "labels": labels
    }
    with open("processed.pickle", mode="wb") as f:
        pickle.dump(save_dict, f)
