from utils import * 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num_threads', default=100)
parser.add_argument('--n_samples', default=1000)
args = parser.parse_args()

local_dir = '/Users/maxchang/Downloads/semantic_annotations'

train_ds, test_ds, validation_ds = parse_into_data_sets(local_dir, args.n_samples, args.num_threads)

