import pickle
import argparse
from sklearn.model_selection import train_test_split
import numpy as np
import time

from finetune import SequenceLabeler, MaskedLanguageModel
from finetune.target_models.semi_suprevised import VATLabeler, PseudoLabeler, MeanTeacherLabeler
from finetune.base_models import RoBERTa, TCN
from finetune.util.metrics import annotation_report, sequence_f1

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
                     (VAT, Pseudo, MLM, Mean).  Unrecognized strs or None
                     trains a Sequence Labeler. (default: None)""")
    parser.add_argument('--base_model',
                     type=str,
                     default=None,
                     help="""What basemodel to train. Current options are
                     (RoBERTa, TCN).  Unrecognized strs or None trains a
                     RoBERTa model. (default: None)""")

    parser.add_argument('--data_usage',
                     type=int,
                     default=100,
                     help="""What percent of the labeled data that will be
                     used as labeled data, should be an int 1 - 100.
                     (default: 100)""")
    parser.add_argument('--epochs',
                     type=int,
                     default=2,
                     help="Epochs to train for. (default: 2)")
    parser.add_argument('--runs',
                     type=int,
                     default=1,
                     help="Runs to average over(default: 1)")
    parser.add_argument('--batch_size',
                     type=int,
                     default=2)
    parser.add_argument('--u_batch_size',
                     type=int,
                     default=4)

    parser.add_argument('--class_weights',
                     type=str2none,
                     default=None,
                     help="""Class weighting to use. Options are (log, linear,
                     sqrt).(default: None)""")
    parser.add_argument('--crf',
                     type=str2bool,
                     default=False,
                     help="Whether or not to use a CRF. (default: False)")
    parser.add_argument('--low_memory',
                     type=str2bool,
                     default=False,
                     help="Whether or not to use low memory mode. (default: False)")
    parser.add_argument('--wandb',
                     type=str2bool,
                     default=False,
                     help="Whether or not to use WandB logging. (default: False)")
    parser.add_argument('--wandb_name',
                     type=str,
                     default=None,
                     help="WandB run name. (default: None)")

    # VAT args
    parser.add_argument('--preturb_embed',
                     type=str2bool,
                     default=True)
    parser.add_argument('--top_k',
                        type=int,
                        default=None)
    parser.add_argument('--k',
                        type=int,
                        default=2)
    parser.add_argument('--e',
                        type=float,
                        default=0.0002)
    parser.add_argument('--loss_coef',
                        type=float,
                        default=0.3)

    # Pseudo args
    parser.add_argument('--thresh',
                        type=float,
                        default=0.99)

    args = parser.parse_args()

    config = dict(
        crf_sequence_labeling = args.crf,
        n_epochs = args.epochs,
        low_memory_mode = args.low_memory,
        batch_size = args.batch_size,
        u_batch_size = args.u_batch_size,
        vat_preturb_embed = args.preturb_embed,
        vat_top_k = args.top_k if (args.top_k and args.top_k > 0) else None,
        vat_k = args.k,
        vat_e = args.e,
        vat_loss_coef = args.loss_coef,
        pseudo_thresh = args.thresh,
        class_weights = args.class_weights
    )

    if args.wandb:
        import wandb
        from wandb.tensorflow import WandbHook
        wandb.init(project="ssl-benchmarks",
                   name=args.wandb_name,
                   sync_tensorboard=True,
                   config=config)
        wandb.config.dataset = args.data.split('/')[-1]
        hooks = WandbHook
    else:
        hooks = None

    filename = args.data
    with open(filename, mode="rb") as f:
        dataset = pickle.load(f)

    allX = dataset["inputs"]
    allY = dataset["labels"]
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

    al = len(allX)
    l = len(trainX)
    p = (l / al) * 100
    print(f"{l} examples labeled of {al} available - {p:.2f}% of the data")
    l = len(unlabeledX)
    p = (l / al) * 100
    print(f"{l} examples unlabeled of {al} available - {p:.2f}% of the data")

    arg_base_model = None if not args.base_model else args.base_model.lower()
    if arg_base_model == "tcn":
        print("TCN selected!")
        base_model = TCN
    else:
        print("RoBERTa selected!")
        base_model = RoBERTa

    algo = None if not args.algo else args.algo.lower()
    if algo is None or algo is "roberta":
        print("Training baseline...")
        model = SequenceLabeler(base_model=base_model,
                                crf_sequence_labeling=args.crf,
                                n_epochs=args.epochs,
                                early_stopping_steps=None,
                                low_memory_mode=args.low_memory)
        model.fit(trainX, trainY, update_hook=hooks)
    elif algo == "vat":
        print("Training VAT...")
        model = VATLabeler(base_model=base_model, **config)
        model.fit(trainX,
                  Us=unlabeledX,
                  Y=trainY,
                  update_hook=hooks)
    elif algo == "pseudo":
        print("Training Pseudo Labels...")
        model = PseudoLabeler(base_model=base_model, **config)
        model.fit(trainX,
                  Us=unlabeledX,
                  Y=trainY,
                  update_hook=hooks)
    elif algo == "mean":
        print("Training Mean Teacher...")
        model = MeanTeacherLabeler(base_model=base_model, **config)
        model.fit(trainX,
                  Us=unlabeledX,
                  Y=trainY,
                  update_hook=hooks)
    elif algo == "mlm":
        print("Training Masked Language Model...")
        class_weights = config["class_weights"]
        config["class_weights"] = None
        model = MaskedLanguageModel(base_model=base_model, **config)
        model.fit(unlabeledX)
        save_file = "bert/ssl_mlm.jl" 
        save_file = save_file + str(int(time.time())) 
        model.create_base_model(save_file)

        model = SequenceLabeler(base_model=base_model,
                                base_model_path=save_file,
                                crf_sequence_labeling=args.crf,
                                n_epochs=args.epochs,
                                early_stopping_steps=None,
                                low_memory_mode=args.low_memory,
                                batch_size=args.batch_size,
                                class_weights=class_weights)
        model.fit(trainX, trainY, update_hook=hooks)


    predictions = model.predict(testX)
    report, averages = annotation_report(testY, predictions)
    print(report)

    f1_token_micro = sequence_f1(testY, predictions, span_type="token", average="micro")
    f1_token_macro = sequence_f1(testY, predictions, span_type="token", average="macro")
    f1_exact_micro = sequence_f1(testY, predictions, span_type="exact", average="micro")
    f1_exact_macro = sequence_f1(testY, predictions, span_type="exact", average="macro")
    if args.wandb:
        wandb.run.summary["Token Micro F1"] = f1_token_micro
        wandb.run.summary["Exact Micro F1"] = f1_exact_micro
        wandb.run.summary["Token Macro F1"] = f1_token_macro
        wandb.run.summary["Exact Macro F1"] = f1_exact_macro
    print(f"Token Micro F1: {f1_token_micro}") 
    print(f"Exact Micro F1: {f1_exact_micro}") 
    print(f"Token Macro F1: {f1_token_macro}") 
    print(f"Exact Macro F1: {f1_exact_macro}") 
