import json
import os
import random
import argparse

random.seed(0)

if __name__ == "__main__":
    dataset_path = 'data/cab/cab.json'
    with open(dataset_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)

    # Shuffle the dataset
    num_samples = len(dataset)            # length of the dataset
    indices = list(range(1,num_samples+1))    # list of indices
    random.shuffle(indices)               # shuffle the indices
    split_point = int(num_samples * 0.7) 
    train_indices = indices[:split_point] # train indices
    test_indices = indices[split_point:]  # test indices
    print(f"Train: {train_indices}, Test: {test_indices}")
    print(f"# Train: {len(train_indices)}, # Test: {len(test_indices)}")

    train_set = {f"example-{i}": dataset[f"example-{i}"] for i in train_indices}
    test_set = {f"example-{i}": dataset[f"example-{i}"] for i in test_indices}
    
    # save train set
    train_path = 'data/cab/cab_train.json'
    with open(train_path, 'w', encoding='utf-8') as train_file:
        json.dump(train_set, train_file, ensure_ascii=False, indent=4)
    
    # save test set
    test_path = 'data/cab/cab_test.json'
    with open(test_path, 'w', encoding='utf-8') as test_file:
        json.dump(test_set, test_file, ensure_ascii=False, indent=4)