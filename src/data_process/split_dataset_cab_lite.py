import json
import os
import random
import argparse
import re

random.seed(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='cab_lite',help='cab_lite')
    args = parser.parse_args()

    if args.dataset != 'cab_lite':
        raise ValueError(f"Unsupported dataset: {args.dataset}. Only 'cab_lite' is supported.")

    dataset_path = f'data/{args.dataset}/{args.dataset}.json'
    with open(dataset_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)

    # Shuffle the dataset
    num_samples = len(dataset)            # length of the dataset
    indices = list(dataset.keys())
    random.shuffle(indices)               # shuffle the indices
    split_point = int(num_samples * 0.7) 
    train_indices = indices[:split_point] # train indices
    test_indices = indices[split_point:]  # test indices
    print(f"Train: {train_indices}, Test: {test_indices}")
    print(f"# Train: {len(train_indices)}, # Test: {len(test_indices)}")
    
    train_set = {i: dataset[i] for i in train_indices}
    test_set = {i: dataset[i] for i in test_indices}
    
    # save train set
    train_path = f'data/{args.dataset}/{args.dataset}_train.json'
    with open(train_path, 'w', encoding='utf-8') as train_file:
        json.dump(train_set, train_file, ensure_ascii=False, indent=4)
    
    # save test set
    test_path = f'data/{args.dataset}/{args.dataset}_test.json'
    with open(test_path, 'w', encoding='utf-8') as test_file:
        json.dump(test_set, test_file, ensure_ascii=False, indent=4)