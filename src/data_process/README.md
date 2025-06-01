# Split dataset
Split the dataset into training and testing.
```shell
bash src/data_process/dataset_split.sh
```

# Generation of SFT data
Running the following script to prepare the json files for each SFT settings.
```shell
bash src/data_process/sft_generate.sh
```
> **We have four settings:**:  1) with think & with personas, 2) without think & with personas, 3) with think & without personas, 4) without think & without personas. 