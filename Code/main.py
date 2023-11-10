import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from data_loader import get_loader




def getimglbllist(config):
	trainfolderlist = sorted([x for x in config.train_path.iterdir() if x.is_dir()])
	eachfolcontent = [None] * len(trainfolderlist)
	survival_data = pd.read_csv(config.survival_csv_path)
	j = 0
	for i, eachfol in enumerate(trainfolderlist):
		temp = sorted([x for x in eachfol.iterdir()])
		if survival_data['Brats20ID'][j] == eachfol.parts[-1]:
			survivaldays = survival_data['Survival_days'][j]
			j += 1
		else:
			survivaldays = None
		eachfolcontent[i] = [{'t2flair':temp[0], 't1':temp[2], 't1ce':temp[3], 't2':temp[4]}, {'mask':temp[1], 'survivaldays':survivaldays}]
	df = pd.DataFrame(eachfolcontent)
	df.columns = ['Images','Labels']
	return df


def main(config):
	df = getimglbllist(config)
	X = df['Images'].tolist()
	y = df['Labels'].tolist()
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)
	train_loader = get_loader(config, X_train, y_train, 'train')
	test_loader = get_loader(config, X_test, y_test, 'test')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# Config parameters
	parser.add_argument('--classes', type=int, default=2)
	parser.add_argument('--num_epochs', type=int, default=30)
	parser.add_argument('--num_epochs_decay', type=int, default=20)
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--lr', type=float, default=0.001)
	parser.add_argument('--mean', type=float, default=[0.485, 0.456, 0.406])
	parser.add_argument('--std', type=float, default=[0.229, 0.224, 0.225])
	parser.add_argument('--train_path', type=str, default=Path('../../Data/MICCAI_BraTS2020_TrainingData'))
	parser.add_argument('--survival_csv_path', type=str, default=Path('../../Data/MICCAI_BraTS2020_TrainingData/survival_info.csv'))
	parser.add_argument('--csv_path', type=str, default=Path('../../Data/filenames.csv'))
	config = parser.parse_args()
	main(config)
