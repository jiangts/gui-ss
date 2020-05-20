from utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num_threads', type=int, default=50)
parser.add_argument('--n_samples', type=int, default=1000)
parser.add_argument('--annotations_path', default='/Users/jiangts/Documents/stanford/cs231n/final_project/semantic_annotations')
parser.add_argument('--images_path', default='/Users/jiangts/Documents/stanford/cs231n/final_project/combined')
args = parser.parse_args()


train_ds, test_ds, validation_ds = parse_into_data_sets(args.annotations_path,
        args.n_samples,
        args.num_threads)

