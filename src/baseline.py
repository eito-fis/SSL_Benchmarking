import pickle
import argparse
from sklearn.model_selection import train_test_split

from finetune import SequenceLabeler
from finetune.target_models.semi_suprevised import SSLLabeler
from finetune.base_models import RoBERTa
from finetune.util.metrics import annotation_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Train SSL')
    parser.add_argument('--algo',
                        type=str,
                        default=None,
                        help="""What algorithm to train. Current options are
                        (VAT).  Unrecognized strs or None trains a default CRF
                        RoBERTa Sequence Labeler. (default: None)""")
    parser.add_argument('--data-usage',
                        type=int,
                        default=100,
                        help="""What percent of the labeled data that will be
                        used as labeled data, should be an int 1 - 100.
                        (default: 100)""")
    args = parser.parse_args()


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

    if args.algo == "VAT":
        model = SSLLabeler(base_model=RoBERTa,
                           crf_sequence_labeling=False)
        model.fit(trainX, Us=unlabeledX, Y=trainY)
    else:
        model = SequenceLabeler(base_model=RoBERTa)
        model.fit(trainX, trainY)

    predictions = model.predict(testX)
    print(annotation_report(testY,
                            predictions))
