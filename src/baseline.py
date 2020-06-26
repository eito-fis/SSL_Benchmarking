import pickle
import argparse
from sklearn.model_selection import train_test_split

from finetune import SequenceLabeler
from finetune.target_models.semi_suprevised import VATLabeler, PseudoLabeler
from finetune.base_models import RoBERTa, TCN
from finetune.util.metrics import annotation_report



if __name__ == "__main__":
    parser = argparse.ArgumentParser('Train SSL')
    parser.add_argument('--algo',
                        type=str,
                        default=None,
                        help="""What algorithm to train. Current options are
                        (VAT, Pseudo).  Unrecognized strs or None trains a
                        default RoBERTa Sequence Labeler. (default: None)""")
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
    parser.add_argument('--wandb',
                        default=False,
                        action='store_true',
                        help="Whether or not to use a CRF. (default: False)")
    args = parser.parse_args()

    if args.wandb:
        import wandb
        from wandb.tensorflow import WandbHook
        wandb.init(project="ssl-benchmarks", sync_tensorboard=True)
        hooks = WandbHook
    else:
        hooks = None


    filename = "data/CONLL-2003/processed.pickle"
    with open(filename, mode="rb") as f:
        dataset = pickle.load(f)
    allX = dataset["inputs"]
    allY = dataset["labels"]
    al = len(allX)
    trainX, testX, trainY, testY = train_test_split(
        allX,
        allY,
        test_size=0.2,
        random_state=40
    )
    split = 1 - args.data_usage / 100
    if split > 0.0:
        trainX, unlabeledX, trainY, _ = train_test_split(
            trainX,
            trainY,
            test_size= 1 - args.data_usage / 100,
            random_state=40
        )
    else:
        unlabeledX = []
    l = len(trainX)
    p = (l / al) * 100
    print(f"{l} examples labeled of {al} available - {p:.2f}% of the data")
    l = len(unlabeledX)
    p = (l / al) * 100
    print(f"{l} examples unlabeled of {al} available - {p:.2f}% of the data")

    base_model = RoBERTa
    if args.algo == "VAT":
        model = VATLabeler(base_model=base_model,
                           crf_sequence_labeling=args.crf,
                           n_epochs=args.epochs,
                           tensorboard_folder="tensorboard/vat",
                           val_interval=2,
                           val_size=0)
        model.fit(trainX, Us=unlabeledX, Y=trainY, update_hook=hooks)
    elif args.algo == "Pseudo":
        model = PseudoLabeler(base_model=base_model,
                              crf_sequence_labeling=args.crf,
                              n_epochs=args.epochs,
                              tensorboard_folder="tensorboard/pseudo",
                              val_interval=2,
                              val_size=0)
        model.fit(trainX, Us=unlabeledX, Y=trainY, update_hook=hooks)
    else:
        model = SequenceLabeler(base_model=base_model,
                                crf_sequence_labeling=args.crf,
                                n_epochs=args.epochs)
        model.fit(trainX, trainY, update_hook=hooks)

    predictions = model.predict(testX)
    print(annotation_report(testY,
                            predictions))
