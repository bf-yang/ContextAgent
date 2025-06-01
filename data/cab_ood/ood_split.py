import json
import random
import os
from collections import defaultdict
# random.seed(10) # 315 test
random.seed(1) # 337 test

# File paths
input_path = "data/cab_ood/cab.json"  # raw cab dataset
train_output = "data/cab_ood/cab_ood_train.json"
test_output = "data/cab_ood/cab_ood_test.json"

# 1. Read JSON file
with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"File read successfully: {input_path}")
print(f"Total samples: {len(data)}")

# 2. Collect all categories and count samples per category
category_counter = defaultdict(int)
for key, item in data.items():
    category = item.get("Category")
    if category:
        category_counter[category] += 1

# 3. Get unique categories
all_categories = list(category_counter.keys())
print(f"\nFound {len(all_categories)} categories:")
for category, count in category_counter.items():
    print(f"  - {category}: {count} samples")

# 4. Split categories: 6 for training, 3 for testing
# Shuffle categories with fixed seed
random.shuffle(all_categories)
train_categories = all_categories[:6]
test_categories = all_categories[6:9]

print("\nCategory split:")
print(f"Training categories ({len(train_categories)}): {', '.join(train_categories)}")
print(f"Testing categories ({len(test_categories)}): {', '.join(test_categories)}")

# 5. Create datasets based on category assignment
train_data = {}
test_data = {}

for key, item in data.items():
    category = item.get("Category")
    if category in train_categories:
        train_data[key] = item
    elif category in test_categories:
        test_data[key] = item

print("\nDataset sizes:")
print(f"Training set: {len(train_data)} samples")
print(f"Testing set: {len(test_data)} samples")

# 6. Save datasets to JSON files
with open(train_output, 'w', encoding='utf-8') as f:
    json.dump(train_data, f, indent=4, ensure_ascii=False)

with open(test_output, 'w', encoding='utf-8') as f:
    json.dump(test_data, f, indent=4, ensure_ascii=False)

print("\nFiles saved:")
print(f"Training set: {os.path.abspath(train_output)}")
print(f"Testing set: {os.path.abspath(test_output)}")

# 7. Verify category distribution in output files
print("\nTraining set category distribution:")
train_counter = defaultdict(int)
for item in train_data.values():
    train_counter[item["Category"]] += 1
for cat, count in train_counter.items():
    print(f"  - {cat}: {count} samples")

print("\nTesting set category distribution:")
test_counter = defaultdict(int)
for item in test_data.values():
    test_counter[item["Category"]] += 1
for cat, count in test_counter.items():
    print(f"  - {cat}: {count} samples")