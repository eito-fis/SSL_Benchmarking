import pickle
from sklearn.model_selection import train_test_split

from finetune import SequenceLabeler
from finetune.base_models import RoBERTa
from finetune.util.metrics import annotation_report


if __name__ == "__main__":
    filename = "data/CONLL-2003/processed.pickle"
    with open(filename, mode="rb") as f:
        dataset = pickle.load(f)
    allX = dataset["inputs"]
    allY = dataset["labels"]
    allX = allX[:2500]
    allY = allY[:2500]
    trainX, testX, trainY, testY = train_test_split(
        allX,
        allY,
        test_size=0.7,
        random_state=40
    )
    model = SequenceLabeler(base_model=RoBERTa,
                            batch_size=1, n_epochs=3,
                            val_size=0.0, max_length=16,
                            chunk_long_sequences=True,
                            subtoken_predictions=True,
                            crf_sequence_labeling=True)
    model.fit(trainX, trainY)
    predictions = model.predict(testX)
    print(predictions)
    print(annotation_report(testY,
                            predictions))
