import pickle
import argparse
from sklearn.model_selection import train_test_split

from finetune import SequenceLabeler
from finetune.target_models.semi_suprevised import VATLabeler, PseudoLabeler
from finetune.base_models import RoBERTa, TCN
from finetune.util.metrics import annotation_report



if __name__ == "__main__":
    parser = argparse.ArgumentParser('Train SSL')
    parser.add_argument('--data',
                        type=str,
                        default="data/CONLL-2003/processed.pickle",
                        help="""Pickle file to load data from. (default:
                        data/CONLL-2003/processed.pickle""")
    parser.add_argument('--algo',
                        type=str,
                        default=None,
                        help="""What algorithm to train. Current options are
                        (VAT, Pseudo).  Unrecognized strs or None trains a
                        Sequence Labeler. (default: None)""")
    parser.add_argument('--base-model',
                        type=str,
                        default=None,
                        help="""What basemodel to train. Current options are
                        (RoBERTa, TCN).  Unrecognized strs or None trains a
                        RoBERTa model. (default: None)""")
    parser.add_argument('--data-usage',
                        type=int,
                        default=100,
                        help="""What percent of the labeled data that will be
                        used as labeled data, should be an int 1 - 100.
                        (default: 100)""")
    parser.add_argument('--epochs',
                        type=int,
                        default=2,
                        help="Epochs to train for. (default: 2)")
    parser.add_argument('--crf',
                        default=False,
                        action='store_true',
                        help="Whether or not to use a CRF. (default: False)")
    parser.add_argument('--low-memory',
                        default=False,
                        action='store_true',
                        help="Whether or not to use low memory mode. (default: False)")
    parser.add_argument('--wandb',
                        default=False,
                        action='store_true',
                        help="Whether or not to use WandB logging. (default: False)")
    args = parser.parse_args()

    if args.wandb:
        import wandb
        from wandb.tensorflow import WandbHook
        wandb.init(project="ssl-benchmarks", sync_tensorboard=True)
        hooks = WandbHook
    else:
        hooks = None


    filename = args.data
    with open(filename, mode="rb") as f:
        dataset = pickle.load(f)
    allX = dataset["inputs"]
    allY = dataset["labels"]
    al = len(allX)
    trainX, testX, trainY, testY = train_test_split(
        allX,
        allY,
        test_size=0.2,
        random_state=42
    )
    split = 1 - args.data_usage / 100
    if split > 0.0:
        trainX, unlabeledX, trainY, _ = train_test_split(
            trainX,
            trainY,
            test_size= 1 - args.data_usage / 100,
            random_state=42
        )
    else:
        unlabeledX = []
    l = len(trainX)
    p = (l / al) * 100
    print(f"{l} examples labeled of {al} available - {p:.2f}% of the data")
    l = len(unlabeledX)
    p = (l / al) * 100
    print(f"{l} examples unlabeled of {al} available - {p:.2f}% of the data")

    arg_base_model = None if not args.base_model else args.base_model.lower()
    if arg_base_model == "tcn":
        base_model = TCN
    else:
        base_model = RoBERTa
    algo = None if not args.algo else args.algo.lower()
    if algo == "vat":
        model = VATLabeler(base_model=base_model,
                           crf_sequence_labeling=args.crf,
                           n_epochs=args.epochs,
                           tensorboard_folder="tensorboard/vat",
                           val_interval=2,
                           val_size=0,
                           low_memory_mode=args.low_memory)
        print("Training VAT...")
        model.fit(trainX, Us=unlabeledX, Y=trainY, update_hook=hooks)
    elif algo == "pseudo":
        model = PseudoLabeler(base_model=base_model,
                              crf_sequence_labeling=args.crf,
                              n_epochs=args.epochs,
                              tensorboard_folder="tensorboard/pseudo",
                              val_interval=2,
                              val_size=0,
                              low_memory_mode=args.low_memory)
        print(model.get_variable_names())
        print("Training Pseudo Labels...")
        model.fit(trainX, Us=unlabeledX, Y=trainY, update_hook=hooks)
    else:
        model = SequenceLabeler(base_model=base_model,
                                crf_sequence_labeling=args.crf,
                                n_epochs=args.epochs,
                                early_stopping_steps=None,
                                low_memory_mode=args.low_memory)
        print("Training baseline...")
        model.fit(trainX, trainY, update_hook=hooks)

    predictions = model.predict(testX)
    print(annotation_report(testY,
                            predictions))
