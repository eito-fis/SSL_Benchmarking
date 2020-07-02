import pickle
import argparse
from sklearn.model_selection import train_test_split
import numpy as np

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
	parser.add_argument('--runs',
					 type=int,
					 default=1,
					 help="Runs to average over(default: 1)")
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
		print("TCN selected!")
		base_model = TCN
	else:
		print("RoBERTa selected!")
		base_model = RoBERTa
		algo = None if not args.algo else args.algo.lower()
		if algo == "vat":
			print("Training VAT...")
			model_fn = lambda: VATLabeler(base_model=base_model,
								 crf_sequence_labeling=args.crf,
								 n_epochs=args.epochs,
								 tensorboard_folder="tensorboard/vat",
								 val_interval=2,
								 val_size=0,
								 low_memory_mode=args.low_memory)
			fit_fn = lambda m: m.fit(trainX, Us=unlabeledX, Y=trainY,
							update_hook=hooks)
		elif algo == "pseudo":
			print("Training Pseudo Labels...")
			model_fn = lambda: PseudoLabeler(base_model=base_model,
									crf_sequence_labeling=args.crf,
									n_epochs=args.epochs,
									tensorboard_folder="tensorboard/pseudo",
									val_interval=2,
									val_size=0,
									low_memory_mode=args.low_memory)
			fit_fn = lambda m: m.fit(trainX, Us=unlabeledX, Y=trainY,
							update_hook=hooks)
		else:
			print("Training baseline...")
			model_fn = lambda: SequenceLabeler(base_model=base_model,
									  crf_sequence_labeling=args.crf,
									  n_epochs=args.epochs,
									  early_stopping_steps=None,
									  low_memory_mode=args.low_memory)
			fit_fn = lambda m: m.fit(trainX, trainY, update_hook=hooks)

	all_averages = []
	for i in range(args.runs):
		print(f"Starting run {i + 1} of {args.runs}...")
		model = model_fn()
		fit_fn(model)
		predictions = model.predict(testX)
		report, averages = annotation_report(testY, predictions)
		print(report)
		predict_dict = {
			"inputs": testX,
			"targets": testY,
			"predictions": predictions,
			"report": report
		}
		print(predictions[:3])
		import json
		class NpEncoder(json.JSONEncoder):
			def default(self, obj):
				if isinstance(obj, np.integer):
					return int(obj)
				elif isinstance(obj, np.floating):
					return float(obj)
				elif isinstance(obj, np.ndarray):
					return obj.tolist()
				else:
					return super(NpEncoder, self).default(obj) 
		with open('prediction_data.json', 'w') as out_file:
			json.dump(predict_dict, out_file, cls=NpEncoder)
			all_averages.append(averages)
		if args.runs > 1:
			all_averages = np.array(all_averages)
			final_averages = np.mean(all_averages, axis=0)
			final_std = np.std(all_averages, axis=0)
			final_max = np.amax(all_averages, axis=0)
			final_min = np.amin(all_averages, axis=0)
			format_fn = lambda l: [f"{x:.2f}" for x in l]
			print(f"Averages: {format_fn(final_averages)}")
			print(f"Standard Deviation: {format_fn(final_std)}")
			print(f"Maxs: {format_fn(final_max)}")
			print(f"Mins: {format_fn(final_min)}")
