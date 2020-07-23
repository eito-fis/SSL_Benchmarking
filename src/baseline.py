import pickle
import argparse
from sklearn.model_selection import train_test_split
import numpy as np
import time

from finetune import SequenceLabeler, MaskedLanguageModel
from finetune.target_models.semi_suprevised import VATLabeler, PseudoLabeler, MeanTeacherLabeler, ICTLabeler
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
    parser.add_argument('--wandb',
                     type=str2bool,
                     default=False,
                     help="Whether or not to use WandB logging. (default: False)")
    parser.add_argument('--wandb_name',
                     type=str,
                     default=None,
                     help="WandB run name. (default: None)")

    parser.add_argument('--algo',
                     type=str,
                     default=None,
                     help="""What algorithm to train. Current options are
                     (vat, pseudo, mlm, mean, ict).  Unrecognized strs or None
                     trains a Sequence Labeler. (default: None)""")
    parser.add_argument('--base_model',
                     type=str,
                     default=None,
                     help="""What basemodel to train. Current options are
                     (RoBERTa, TCN).  Unrecognized strs or None trains a
                     RoBERTa model. (default: None)""")

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
    parser.add_argument('--iterate_unlabeled',
                     type=str2bool,
                     default=True,
                     help="""Whether or not to count epochs over unlabeled data.
                     (default: True)""")
    parser.add_argument('--tsa_method',
                     type=str2none,
                     default=None)

    parser.add_argument('--labeled_percent',
                     type=int,
                     default=None,
                     help="""What percent of the labeled data that will be
                     used as labeled data, should be an int 1 - 100. 
                     (default: 200 labeled examples)""")
    parser.add_argument('--labeled_count',
                     type=int,
                     default=200,
                     help="""The number of labeled examples to be used.
                     (default: 200 labeled examples)""")
    parser.add_argument('--epochs',
                     type=int,
                     default=2,
                     help="Epochs to train for. (default: 2)")
    parser.add_argument('--train_steps',
                     type=int,
                     default=None,
                     help="""The amount of training steps to take. Takes
                        precedent over epochs if set. (default: None)""")
    parser.add_argument("--batch_ratio",
                     type=float,
                     default=None)


    parser.add_argument('--batch_size',
                     type=int,
                     default=2)
    parser.add_argument('--u_batch_size',
                     type=int,
                     default=4)
    parser.add_argument('--loss_coef',
                     type=float,
                     default=0.3)
    parser.add_argument('--decay',
                     type=float,
                     default=0.999)

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

    # Pseudo args
    parser.add_argument('--thresh',
                     type=float,
                     default=0.99)
    # ICT args
    parser.add_argument('--alpha',
                     type=float,
                     default=0.2)

    args = parser.parse_args()



    algo = None if not args.algo else args.algo.lower()
    config = dict(
        crf_sequence_labeling = args.crf,
        n_epochs = args.epochs,
        low_memory_mode = args.low_memory,
        iterate_unlabeled=args.iterate_unlabeled,
        batch_size = args.batch_size,
        u_batch_size = args.u_batch_size,
        ssl_loss_coef = args.loss_coef,
        ema_decay = args.decay,
        vat_preturb_embed = args.preturb_embed,
        vat_top_k = args.top_k if (args.top_k and args.top_k > 0) else None,
        vat_k = args.k,
        vat_e = args.e,
        pseudo_thresh = args.thresh,
        ict_alpha = args.alpha,
        class_weights = args.class_weights,
        tsa_method=args.tsa_method,
        val_size=0,
        train_embeddings=True,
    )

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

    if args.labeled_percent is None:
        data_usage = (args.labeled_count / len(trainX)) * 100
    else:
        dat_usage = args.labled_percent

    split = 1 - data_usage / 100
    if split > 0.0:
        trainX, unlabeledX, trainY, _ = train_test_split(
            trainX,
            trainY,
            test_size=split,
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

    # Adjust epochs to the amount of epochs an SSL approach would see
    if algo is None or algo == "roberta":
        if args.batch_ratio:
            # How many suprevised epochs there are in an unsuprevised epoch
            # assuming 1:1 batch ratio
            u_s_len_ratio = len(unlabeledX) / len(trainX)
            # How many suprevised epochs there are in an unsuprevised epoch,
            # accounting for different batch_ratios
            epoch_multiplier = u_s_len_ratio * args.batch_ratio
            config["n_epochs"] = round(config["n_epochs"] * epoch_multiplier)

    # If train steps are passed, convert to epochs for finetune
    if args.train_steps:
        if args.iterate_unlabeled:
            steps_per_epoch = len(unlabeledX)
        else:
            steps_per_epoch = len(trainX)
        n_epochs = round(args.train_steps / steps_per_epoch)
        real_train_steps = int(steps_per_epoch) * n_epochs
        print(f"{args.train_steps} trains steps became {n_epochs} epochs")
        print(f"Actual train steps: {real_train_steps}")
        config["n_epochs"] = n_epochs

    if args.wandb:
        import wandb
        from wandb.tensorflow import WandbHook
        wandb.init(project="ssl-benchmarks",
             name=args.wandb_name,
             sync_tensorboard=True,
             config=config)
        wandb.config.dataset = args.data.split('/')[-1]
        wandb.config.batch_ratio = args.batch_ratio
        wandb.config.data_usage = data_usage
        wandb.config.algo = algo.lower()
        if args.train_steps:
            wandb.config.train_steps = args.train_steps
            wandb.config.real_train_steps = real_train_steps
        hooks = WandbHook
    else:
        hooks = None


    arg_base_model = None if not args.base_model else args.base_model.lower()
    if arg_base_model == "tcn":
        print("TCN selected!")
        base_model = TCN
    else:
        print("RoBERTa selected!")
        base_model = RoBERTa

    algo_2_model = {
        "vat": VATLabeler,
        "pseudo": PseudoLabeler,
        "mean": MeanTeacherLabeler,
        "ict": ICTLabeler
    }
    if algo in algo_2_model:
        model = algo_2_model[algo](base_model=base_model, **config,
                                   tensorboard_folder="tensorboard/testing",
                                   val_interval=1)
        model.fit(trainX,
                  Us=unlabeledX,
                  Y=trainY,
                  update_hook=hooks)
    elif algo is None:
        print("Training baseline...")
        model = SequenceLabeler(base_model=base_model,
                          crf_sequence_labeling=config["crf_sequence_labeling"],
                          n_epochs=config["n_epochs"],
                          batch_size=config["batch_size"],
                          class_weights=config["class_weights"],
                          early_stopping_steps=None,
                          tensorboard_folder="tensorboard/testing",
                          val_interval=1,
                          val_size=0,
                          low_memory_mode=config["low_memory_mode"])
        model.fit(trainX, trainY, update_hook=hooks)
    elif algo == "mlm":
        print("Training Masked Language Model...")
        class_weights = config["class_weights"]
        config["class_weights"] = None
        config["iterate_unlabeled"] = False
        model = MaskedLanguageModel(base_model=base_model, **config)
        mlmX = unlabeledX + trainX
        model.fit(mlmX)
        save_file = "bert/ssl_mlm.jl" 
        save_file = save_file + str(int(time.time())) 
        model.create_base_model(save_file)

        model = SequenceLabeler(base_model=base_model,
                                base_model_path=save_file,
                                crf_sequence_labeling=args.crf,
                                batch_size=20,
                                n_epochs=250,
                                early_stopping_steps=None,
                                low_memory_mode=args.low_memory,
                                class_weights=class_weights)
        model.fit(trainX, trainY, update_hook=hooks)


    predictions = model.predict(testX)
    report, averages = annotation_report(testY, predictions)
    print(report)

    f1_token_micro = sequence_f1(testY, predictions, span_type="token", average="micro")
    f1_token_macro = sequence_f1(testY, predictions, span_type="token", average="macro")
    f1_exact_micro = sequence_f1(testY, predictions, span_type="exact", average="micro")
    f1_exact_macro = sequence_f1(testY, predictions, span_type="exact", average="macro")
    print(f"Token Micro F1: {f1_token_micro}") 
    print(f"Exact Micro F1: {f1_exact_micro}") 
    print(f"Token Macro F1: {f1_token_macro}") 
    print(f"Exact Macro F1: {f1_exact_macro}") 
    if args.wandb:
        wandb.run.summary["Token Micro F1"] = f1_token_micro
        wandb.run.summary["Exact Micro F1"] = f1_exact_micro
        wandb.run.summary["Token Macro F1"] = f1_token_macro
        wandb.run.summary["Exact Macro F1"] = f1_exact_macro
