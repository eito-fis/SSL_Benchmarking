import pickle
import argparse
from sklearn.model_selection import train_test_split
import numpy as np
import time
from finetune.base_models.huggingface.models import HFT5
from finetune.target_models.seq2seq import HFS2S
import textdistance
import re

from finetune import SequenceLabeler, MaskedLanguageModel
from finetune.base_models import RoBERTa, TCN
from finetune.util.metrics import annotation_report, sequence_f1

METRIC_FUNCS = {}
def metric_func(name):
    def wrapper(func):
        METRIC_FUNCS[name] = func
        return func
    return wrapper

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2none(v):
    if v is None or v.lower() == "none":
        return None
    return v
@metric_func("reconstruction")
def get_reconstruction_metrics(preds, labels, **kwargs):
    distances = []
    similarities = []
    for pred, label in zip(preds, labels):
        distances.append(textdistance.levenshtein.normalized_distance(pred, label))
        similarities.append(textdistance.levenshtein.normalized_similarity(pred, label))
    distances = np.array(distances)
    similarities = np.array(similarities)
    ret_dict = {
        "Mean Distance": np.mean(distances),
        "Median Distance": np.median(distances),
        "Std Deviation Distance": np.std(distances),
        "Mean Similarity": np.mean(similarities),
        "Median Similarity": np.median(similarities),
        "Std Deviation Similarity": np.std(similarities),
    }
    return ret_dict, None

@metric_func("extraction")
def get_extraction_metrics(preds, labels, delim=" | ", **kwargs):
    true_pos, false_pos, false_neg, total = 0, 0, 0, 0
    error_indicies = []
    for i, (pred, label) in enumerate(zip(preds, labels)):
        # Normalize punctuation
        label = re.sub(" ([.?!:;,])", "\\1", label)
        pred = re.sub(" ([.?!:;,])", "\\1", pred)

        set_labels = set(label.split(delim))
        set_preds = set(pred.split(delim))

        # True postive = tokens in labels and preds
        # False positive = tokens in preds but not in labels
        # False negative = tokens in labels but not in preds
        overlap = set_labels & set_preds
        only_preds = set_preds - overlap
        only_labels = set_labels - overlap
        true_pos += len(overlap)
        false_pos += len(only_preds)
        false_neg += len(only_labels)

        if len(only_preds) > 0 or len(only_labels) > 0:
            error_indicies.append(i)

    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    f1 = 2 * (precision * recall) / (precision + recall)
    ret_dict = {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }
    return ret_dict, error_indicies

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Train SSL')
    parser.add_argument('--data',
                     type=str,
                     default="data/CONLL-2003/processed.pickle",
                     help="""Pickle file to load data from. (default:
                     data/CONLL-2003/processed.pickle""")
    parser.add_argument('--wandb',
                     type=str2bool,
                     default=False,
                     help="Whether or not to use WandB logging. (default: False)")
    parser.add_argument('--wandb_name',
                     type=str,
                     default=None,
                     help="WandB run name. (default: None)")

    parser.add_argument('--base_model',
                     type=str,
                     default="roberta",
                     help="""What basemodel to train. Current options are
                     (RoBERTa, T5).  Unrecognized strs or None trains a
                     RoBERTa model. (default: None)""")

    parser.add_argument('--class_weights',
                     type=str2none,
                     default=None,
                     help="""Class weighting to use. Options are (log, linear,
                     sqrt).(default: None)""")
    parser.add_argument('--crf',
                     type=str2bool,
                     default=True,
                     help="Whether or not to use a CRF. (default: True)")
    parser.add_argument('--low_memory',
                     type=str2bool,
                     default=False,
                     help="Whether or not to use low memory mode. (default: False)")
    parser.add_argument('--train',
                     type=str2bool,
                     default=True,
                     help="Whether or not to train. (default: True)")

    parser.add_argument('--epochs',
                     type=int,
                     default=2,
                     help="Epochs to train for. (default: 2)")
    parser.add_argument('--train_steps',
                     type=int,
                     default=None,
                     help="""The amount of training steps to take. Takes
                        precedent over epochs if set. (default: None)""")
    parser.add_argument('--load',
                        type=str,
                        default=None)
    parser.add_argument('--save',
                        type=str,
                        default=None)
    parser.add_argument('--errors',
                        type=str2bool,
                        default=True)
    parser.add_argument('--mode',
                        type=str,
                        default="extraction")

    parser.add_argument('--labeled_count',
                     type=int,
                     default=None,
                     help="""The number of labeled examples to be used.
                     (default: None)""")


    parser.add_argument('--batch_size',
                     type=int,
                     default=2)
    parser.add_argument('--beam_size',
                     type=int,
                     default=1)
    args = parser.parse_args()



    config = dict(
        crf_sequence_labeling = args.crf,
        n_epochs = args.epochs,
        low_memory_mode = args.low_memory,
        batch_size = args.batch_size,
        class_weights = args.class_weights,
        val_size=0,
        val_interval=1,
        train_embeddings=True,
        tensorboard_folder="tensorboard/text_gen/",
        early_stopping_steps=None,
        beam_size=args.beam_size,
        # permit_uninitialized=".*",
    )

    filename = args.data
    with open(filename, mode="rb") as f:
        dataset = pickle.load(f)

    trainX = dataset["inputs"]
    trainY = dataset["labels"]
    testX = dataset.get("val_inputs")
    testY = dataset.get("val_targets")
    
    if testX and testY:
        print("Validation set found!")
    else:
        trainX, testX, trainY, testY = train_test_split(
            trainX,
            trainY,
            test_size=0.2,
            random_state=42
        )

    # If number of labeled examples are specific, cut down the dataset to size
    if args.labeled_count:
        data_usage = (args.labeled_count / len(trainX)) * 100
        split = 1 - data_usage / 100
        if split > 0.0:
            trainX, _, trainY, _ = train_test_split(
                trainX,
                trainY,
                test_size=split,
                random_state=42
            )
        testX = trainX
        testY = trainY

    # If train steps are passed, convert to epochs for finetune
    if args.train_steps:
        steps_per_epoch = len(trainX)
        n_epochs = round(args.train_steps / steps_per_epoch)
        real_train_steps = int(steps_per_epoch) * n_epochs
        print(f"{args.train_steps} trains steps became {n_epochs} epochs")
        print(f"Actual train steps: {real_train_steps}")
        config["n_epochs"] = n_epochs

    if args.wandb:
        import wandb
        from wandb.tensorflow import WandbHook
        wandb.init(project="text_generation",
             name=args.wandb_name,
             sync_tensorboard=True,
             config=config)
        wandb.config.dataset = args.data.split('/')[-1]
        wandb.config.basemodel = args.base_model
        if args.train_steps:
            wandb.config.train_steps = args.train_steps
            wandb.config.real_train_steps = real_train_steps
        hooks = WandbHook
    else:
        hooks = None


    arg_base_model = None if not args.base_model else args.base_model.lower()
    if arg_base_model == "t5":
        print("T5 selected!")
        base_model = HFT5
        algo = HFS2S
    else:
        print("RoBERTa selected!")
        base_model = RoBERTa
        algo = SequenceLabeler

    if args.load:
        model = algo.load(args.load)
        print(f"Model loaded from {args.load}")
    else:
        model = algo(base_model=base_model, **config)
        print("Model built!")
    if args.train:
        print("Training...")
        model.fit(trainX, trainY, update_hook=hooks)
    if args.save:
        model.save(args.save)
        print(f"Model saved to {args.save}")


    predictions = model.predict(testX)
    if arg_base_model == "roberta":
        # def process_preds(preds):
        #     text = [p["text"].split(" ") for p in preds]
        #     flat_text = [_t for t in text for _t in t]
        #     return " | ".join(flat_text)
        from nameparser import HumanName
        def process_preds(labels):
            _l = len(labels)
            i = 0
            _names = []
            while (i < len(labels)):
                if labels[i]["label"] == "B-PER":
                    start = i
                    i += 1
                    while (i < len(labels) and labels[i]["label"] == "I-PER"):
                        i += 1
                    sub_name = " ".join([l["text"] for l in labels[start:i]])
                    name = HumanName(sub_name)
                    if name.first:
                        if name.last:
                            _names.append(", ".join((name.last, name.first)))
                        else:
                            _names.append(name.first)
                else:
                    i += 1
            concat = " | ".join(_names)
            return concat

        predictions = list(map(process_preds, predictions))
        testY = list(map(process_preds, testY))

    cut = 20
    print(f"FIRST {cut} EXAMPLES")
    print("=" * 40)
    for t, p, a in zip(testX[:cut], predictions[:cut], testY[:cut]):
        print("Text: ", t)
        print("-")
        print("Predictions: ", p)
        print("-")
        print("Label: ", a)
        print("==" * 20)

    metrics, error_indicies = METRIC_FUNCS[mode](preds, labels, delim=" | ")
    print("\n\n")
    for key, value in metrics.items():
        print(f"{metric}: {value}")
        if args.wandb:
            wandb.run.summary[key] = value
    print("\n\n")

    if args.errors:
        if error_indicies:
            print(f"{len(error_indicies)} ERRORS")
            print("=" * 40)
        else:
            error_indicies = range(len(testX))
        step = None
        for i in error_indicies:
            print("Text: ", testX[i])
            print("-")
            print("Predictions: ", predictions[i])
            print("-")
            print("Label: ", testY[i])
            print("==" * 20)
            if not step:
                step = input()
