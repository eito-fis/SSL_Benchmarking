import pickle
import argparse
from sklearn.model_selection import train_test_split
import numpy as np
import time
from finetune.base_models.huggingface.models import HFT5
from finetune.target_models.seq2seq import HFS2S
import textdistance
import re
from itertools import permutations

from finetune import SequenceLabeler, MaskedLanguageModel
from finetune.base_models import RoBERTa, TCN
from finetune.util.metrics import annotation_report, sequence_f1

from util import METRIC_FUNCS, ASSOCIATION_FUNCS, PROCESS_RULES

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

def nice_print(i, p, l):
    print("Text: ", i)
    print("-" * 20)
    print("Predictions: ", p)
    print("-" * 20)
    print("Label: ", l)
    print("==" * 20)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Train SSL')
    parser.add_argument('--data',
                     type=str,
                     default="data/CONLL-2003/processed.pickle",
                     help="""Pickle file to load data from. (default:
                     data/CONLL-2003/processed.pickle""")
    parser.add_argument('--cached_predict',
                     type=str,
                     default=None)
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
    parser.add_argument('--association_mode',
                        type=str,
                        default=None)
    parser.add_argument('--extra_delim',
                        type=str,
                        default=None)

    parser.add_argument('--batch_size',
                     type=int,
                     default=2)
    parser.add_argument('--predict_batch_size',
                     type=int,
                     default=20)
    parser.add_argument('--xla',
                     type=str2bool,
                     default=False)
    parser.add_argument('--beam_size',
                     type=int,
                     default=1)
    parser.add_argument('--beam_alpha',
                     type=float,
                     default=0.2)
    args = parser.parse_args()

    delim_tokens = None
    if args.extra_delim:
        # TODO: Make this a constant tied to dataset in another file
        delim_tokens = ["|", "ref-marker", "authors", "title", "venue", "data",
                        "reference-id", "note", "web", "status", "language",
                        "booktitle", "date", "address", "pages", "organization",
                        "volume", "number", "pubisher", "editor", "tech",
                        "institution", "series", "chapter", "thesis", "school",
                        "department", "person", "person-first", "person-last",
                        "person-middle", "person-affix", "year", "month", "journal"]
        delim_tokens.append(args.extra_delim.split(" "))

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
        beam_search_alpha=args.beam_alpha,
        predict_batch_size=args.predict_batch_size,
        xla=args.xla,
        delim_tokens=delim_tokens,
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

    if not args.cached_predict:
        if args.load:
            model = algo.load(args.load,
                              predict_batch_size=args.predict_batch_size,
                              xla=args.xla,
                              delim_tokens=delim_tokens,
                              beam_size=args.beam_size,
                              beam_search_alpha=args.beam_alpha)
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
        save_dict = {
            "pred": predictions,
            "textY": testY
        }
        predict_filename = "data/predictions/predictions.pickle"
        with open(predict_filename, "wb") as f:
            pickle.dump(save_dict, f)
        print(f"Predictions cached at {predict_filename}!")
    else:
        with open(args.cached_predict, "rb") as f:
            save_dict = pickle.load(f)
            predictions, testY = save_dict["pred"], save_dict["textY"]

    replacements = PROCESS_RULES[args.association_mode]
    for repl in replacements:
        process_fn = lambda x: re.sub(repl[0], repl[1], x)
        predictions = list(map(process_fn, predictions))
        testY = list(map(process_fn, testY))

    cut = 20
    print("HEAD\n" + "=" * 40)
    for i, p, l in zip(testX[:cut], predictions[:cut], testY[:cut]):
        nice_print(i, p, l)

    metrics, error_indicies = METRIC_FUNCS[args.mode](predictions, testY,
                                                      delim="\\|",
                                                      association_mode=args.association_mode)
    print("\n\n")
    for key, value in metrics.items():
        print(f"{key}: {value}")
        if args.wandb:
            wandb.run.summary[key] = value
    print("\n\n")

    if args.errors:
        if error_indicies:
            print(f"{len(error_indicies)} ERRORS\n" + "=" * 40)
        else:
            error_indicies = range(len(testX))
        step = None
        for i in error_indicies:
            nice_print(testX[i], predictions[i], testY[i])
            if not step:
                step = input()
