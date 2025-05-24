# -*- coding: utf-8 -*-
# Evaluation metrics
import json
import os
from typing import Sequence
from openai import OpenAI
from transformers.utils.versions import require_version
require_version("openai>=1.5.0", "To fix: pip install openai>=1.5.0")
import json
import sys 
sys.path.append("/home/bufang/ProAgent/src") 
from functions import *
import ollama
from openai import AzureOpenAI
from utils import azure_inference,ollama_inference,\
calculate_set_metrics,calculate_accuracy,calculate_regression_metrics
import argparse
import re
from tqdm import tqdm
import csv
import ast
import transformers
transformers.logging.set_verbosity_error()
import numpy as np
from collections import Counter

api_key = "4d2ff10a8c3d4d09883a4411832b6718" # Azure API key
client = AzureOpenAI(
    api_key = api_key,  
    api_version = "2023-05-15",
    azure_endpoint = "https://cuhk-aiot-gpt4.openai.azure.com/"
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='cab',help='cab, cab_lite, cab_ood')
    parser.add_argument("--methods", type=str, default='icl',help='icl,sft')
    parser.add_argument("--model_base_sft", type=str, default='llama8b',
                    help='base SFT model: qwen7b, llama8b')
    parser.add_argument("--model_base_icl", type=str, default='qwen2.5:latest', 
                        help='qwen2.5:latest,qwen2.5:72b,gpt-4o')
    parser.add_argument("--zs", type=str, default='false',help='false, true')
    parser.add_argument("--think", type=str, default='w_t',help='w_t, wo_t')
    parser.add_argument("--personas", type=str, default='w_p',help='w_p, wo_p')
    args = parser.parse_args()

    # Read model prediction files
    dataset_name = args.dataset # dataset
    methods = args.methods      # icl or sft
    model_base_icl = args.model_base_icl # qwen2.5:72b,gpt-4o
    model_base_sft = args.model_base_sft # qwen7b, llama8b

    if methods == 'icl':
        pred_path = f'results/{dataset_name}/predictions/{methods}/pred_{model_base_icl}_fs_10_{args.personas}_{args.think}.json'
    elif methods == 'sft':
        pred_path = f'results/{dataset_name}/predictions/{methods}/pred_{model_base_sft}_{args.personas}_{args.think}.json'

    print("Prediction Path: ", pred_path)
    with open(pred_path, 'r') as f:
        dataset = json.load(f)

    gt_thoughts_list,pred_thoughts_list = [],[]
    gt_proactive_idx_list,pred_proactive_idx_list = [],[]
    gt_proactive_score_list,pred_proactive_score_list = [],[]
    gt_actions_list,pred_actions_list = [],[]
    gt_tools_list,pred_tools_list = [],[]
    P_tool_names_list, R_tool_names_list, F1_tool_names_list = [],[],[]
    # Initialize the metrics for each level
    level_metrics = {
        "level1": {"gt_tools_list": [], "pred_tools_list": [], "gt_proactive_idx_list": [], "pred_proactive_idx_list": [], "gt_proactive_score_list": [], "pred_proactive_score_list": [], "P_tool_names_list": [], "R_tool_names_list": [], "F1_tool_names_list": [], "arg_false":0, "arg_counts":0},
        "level2": {"gt_tools_list": [], "pred_tools_list": [], "gt_proactive_idx_list": [], "pred_proactive_idx_list": [], "gt_proactive_score_list": [], "pred_proactive_score_list": [], "P_tool_names_list": [], "R_tool_names_list": [], "F1_tool_names_list": [], "arg_false":0, "arg_counts":0},
        "level3": {"gt_tools_list": [], "pred_tools_list": [], "gt_proactive_idx_list": [], "pred_proactive_idx_list": [], "gt_proactive_score_list": [], "pred_proactive_score_list": [], "P_tool_names_list": [], "R_tool_names_list": [], "F1_tool_names_list": [], "arg_false":0, "arg_counts":0},
    }

    level_indices = {
        "level1": [],
        "level2": [],
        "level3": []
    }
    level_proactive = []

    arg_false, arg_counts = 0, 0
    cnt,cnt_f = 0,0
    tool_level_list = []
    for idx, key in enumerate(tqdm(dataset.keys())):
        data = dataset[key]
        # GT
        gt_thoughts, gt_proactive_idx, gt_proactive_score, gt_actions, gt_tools, gt_persona = data['Thoughts'], data['Proactive index'], data['Proactive score'], data['Action'], data['Tools'], data['Personas']

        # Predictions
        preditions = data["predictions"]
        pred_thoughts, pred_proactive_idx, pred_proactive_score, pred_actions, pred_tools, pred_response = preditions['thoughts'], preditions['proactive_idx'], preditions['proactive_score'], preditions['actions'], preditions['tools'], preditions['response']
            
        # Calculate GPT-score
        gt_proactive_idx = str(gt_proactive_idx)
        gt_proactive_idx = gt_proactive_idx[0].lower() + gt_proactive_idx[1:]
        pred_proactive_idx = pred_proactive_idx[0].lower() + pred_proactive_idx[1:]

        # Replace "medium" with "true"
        if gt_proactive_idx == "medium":
            gt_proactive_idx = "true"

        # extract tool names from json to list
        if pred_tools == 'None' and gt_tools == 'None': # Case for None proactiveness
            pred_tool_names = ['None']
            gt_tool_names = ['None']
        else:
            pred_tool_names = []
            gt_tool_names = []

        # predicted tools
        try:
            json_tool = ast.literal_eval(pred_tools)
            if isinstance(json_tool, list) and pred_tools != 'None':
                for tool in json_tool:
                    if isinstance(tool, dict) and "name" in tool:
                        pred_tool_names.append(tool["name"])
                    else:
                        pred_tool_names = ['Error']
                        print(f"Unexpected tool format: {tool}")
        except (SyntaxError, ValueError) as e:
            print(f"Error parsing pred_tools: {e}")
            pred_tool_names = ['Error']
        pred_tool_names_set = set(pred_tool_names)
        
        # GT tools
        json_tool = ast.literal_eval(gt_tools)
        if gt_tools != 'None':
            for tool in json_tool:
                gt_tool_names.append(tool["name"])
        gt_tool_names_set = set(gt_tool_names)

        # Check the number of tools
        tool_level_list.append(len(gt_tool_names_set))
        if len(gt_tool_names_set) in [0, 1]:
            level = "level1"
        if len(gt_tool_names_set) in [2]:
            level = "level2"
        if len(gt_tool_names_set) in [3, 4]:
            level = "level3"
        level_indices[level].append(key)

        P_tool_name,R_tool_name,F1_tool_name = calculate_set_metrics(pred_tool_names_set, gt_tool_names_set, "Tool Names")
        P_tool_names_list.append(P_tool_name)
        R_tool_names_list.append(R_tool_name)
        F1_tool_names_list.append(F1_tool_name)
        
        if gt_proactive_idx == 'false':
            cnt_f += 1

        # Modify the data format of GT to string 
        # Convert boolean to string
        gt_proactive_idx = "true" if gt_proactive_idx else "false"
        # Convert int to string
        gt_proactive_score = str(gt_proactive_score)

        # Theshold to determine the proactive index
        if int(gt_proactive_score) >= 3:
            gt_proactive_idx = "true"
        else:
            gt_proactive_idx = "false"

        if pred_proactive_score!='None' and int(pred_proactive_score) >= 3:
            pred_proactive_idx = "true"
        else:
            pred_proactive_idx = "false"

        if gt_proactive_idx == 'true' and pred_proactive_idx == 'false':
            arg_false += 1
            arg_counts += 1

        # Check if the tool result contains "error"
        if preditions['tools_results'] != 'None':
            for tool_result in preditions['tools_results']:
                # If the predicted tool is not in the GT tools, it is considered an error
                if tool_result.get("tool_name") not in gt_tool_names_set:
                    arg_false += 1
                else:
                    # If the tool is correct, check if the parameters are incorrect
                    if "error" in tool_result.get("results", ""):
                        arg_false += 1
                arg_counts += 1

        # append to list
        gt_thoughts_list.append(gt_thoughts)
        pred_thoughts_list.append(pred_thoughts)
        gt_proactive_idx_list.append(gt_proactive_idx)
        pred_proactive_idx_list.append(pred_proactive_idx)
        gt_proactive_score_list.append(gt_proactive_score)
        pred_proactive_score_list.append(pred_proactive_score)
        gt_actions_list.append(gt_actions)      
        pred_actions_list.append(pred_actions)
        gt_tools_list.append(gt_tools)
        pred_tools_list.append(pred_tools)
        
        # Add the results to the corresponding level
        level_metrics[level]["gt_tools_list"].append(gt_tools)
        level_metrics[level]["pred_tools_list"].append(pred_tools)
        level_metrics[level]["gt_proactive_idx_list"].append(gt_proactive_idx)
        level_metrics[level]["pred_proactive_idx_list"].append(pred_proactive_idx)
        level_metrics[level]["gt_proactive_score_list"].append(gt_proactive_score)
        level_metrics[level]["pred_proactive_score_list"].append(pred_proactive_score)
        level_metrics[level]["P_tool_names_list"].append(P_tool_name)
        level_metrics[level]["R_tool_names_list"].append(R_tool_name)
        level_metrics[level]["F1_tool_names_list"].append(F1_tool_name)
        level_metrics[level]["arg_false"] = arg_false
        level_metrics[level]["arg_counts"] = arg_counts

    def evaluate_level(level_name, metrics):
        print(f"Evaluating {level_name}")
        print("=" * 50)

        # Tool Use BERTScore
        print(f"Average Precision for Tool Names: {np.mean(metrics[level_name]['P_tool_names_list']):.4f}")
        print(f"Average Recall for Tool Names: {np.mean(metrics[level_name]['R_tool_names_list']):.4f}")
        print(f"Average F1 for Tool Names: {np.mean(metrics[level_name]['F1_tool_names_list']):.4f}")

        arg_acc = 1 - (metrics[level_name]["arg_false"] / metrics[level_name]["arg_counts"]) if metrics[level_name]["arg_counts"] > 0 else 0
        print(f"Argument Accuracy: {arg_acc*100:.2f}%")
        
        # Proactive Index Acc
        print(50*"=")
        acc_idx, incorrect_proactive_idx, miss_needed, false_detection = calculate_accuracy(metrics[level_name]['pred_proactive_idx_list'], metrics[level_name]['gt_proactive_idx_list'], "Proactive Index")
        print(f"Accuracy for Proactive Index: {acc_idx:.2%}")
        print(f"Incorrect indices for Proactive Index: {incorrect_proactive_idx}")
        print(f"Miss-Needed Rate: {miss_needed:.2%}")
        print(f"False-Detection Rate: {false_detection:.2%}")

        # Proactive Score Acc
        print(50*"=")
        acc_score, incorrect_proactive_score, _, _  = calculate_accuracy(metrics[level_name]['pred_proactive_score_list'], metrics[level_name]['gt_proactive_score_list'], "Proactive Score")
        print(f"Accuracy for Proactive Score: {acc_score:.2%}")
        print(f"Incorrect indices for Proactive Score: {incorrect_proactive_score}")

        # Proactive Score RMSE
        metrics[level_name]['pred_proactive_score_list'] = ['0' if score == 'None' else score for score in metrics[level_name]['pred_proactive_score_list']] # replace "None" with 0
        print(50*"=")
        gt_proactive_score_list = list(map(float, metrics[level_name]['gt_proactive_score_list']))
        pred_proactive_score_list = list(map(float, metrics[level_name]['pred_proactive_score_list']))
        mse, rmse = calculate_regression_metrics(pred_proactive_score_list, gt_proactive_score_list, "Proactive Score")   
        print(f"RMSE for Proactive Score: {rmse:.4f}")
        print(50*"=")

    for level in level_metrics.keys():
        print("\033[94m" + 50*"*" + "\033[0m")
        evaluate_level(level, level_metrics)


    # Tool Use BERTScore
    print("\033[94m" + 50*"*" + "\033[0m")
    print("Evaluating Overall")
    print(50*"=")
    print(f"Average Precision for Tool Names: {np.mean(P_tool_names_list):.4f}")
    print(f"Average Recall for Tool Names: {np.mean(R_tool_names_list):.4f}")
    print(f"Average F1 for Tool Names: {np.mean(F1_tool_names_list):.4f}")
    # Calcluate argument accuracy
    arg_acc = 1 - (arg_false / arg_counts)
    print(f"Argument Accuracy: {arg_acc*100:.2f}%")

    # Proactive Index Acc
    print(50*"=")
    acc_idx, incorrect_proactive_idx, miss_needed, false_detection = calculate_accuracy(pred_proactive_idx_list, gt_proactive_idx_list, "Proactive Index")
    print(f"Accuracy for Proactive Index: {acc_idx:.2%}")
    print(f"Incorrect indices for Proactive Index: {incorrect_proactive_idx}")
    print(f"Miss-Needed Rate: {miss_needed:.2%}")
    print(f"False-Detection Rate: {false_detection:.2%}")
    
    # Proactive Score Acc
    print(50*"=")
    acc_score, incorrect_proactive_score, _, _ = calculate_accuracy(pred_proactive_score_list, gt_proactive_score_list, "Proactive Score")
    print(f"Accuracy for Proactive Score: {acc_score:.2%}")
    print(f"Incorrect indices for Proactive Score: {incorrect_proactive_score}")
    
    # Proactive Score RMSE
    pred_proactive_score_list # some predictions can be "None"
    pred_proactive_score_list = ['0' if score == 'None' else score for score in pred_proactive_score_list] # replace "None" with 0
    print(50*"=")
    gt_proactive_score_list = list(map(float, gt_proactive_score_list))
    pred_proactive_score_list = list(map(float, pred_proactive_score_list))
    mse, rmse = calculate_regression_metrics(pred_proactive_score_list, gt_proactive_score_list, "Proactive Score")   
    print(f"RMSE for Proactive Score: {rmse:.4f}")
    print(50*"=")


    print("\033[94m" + 50*"*" + "\033[0m")
    print(f"Total number of examples: {cnt + cnt_f}")
    print(f"Number of proactive examples: {cnt}")
    print(f"Number of non-proactive examples: {cnt_f}")

    gt_tools_counter = Counter(tool_level_list)
    print("GT Tools Number Distribution:")
    for tool, count in gt_tools_counter.items():
        print(f"{tool}: {count}")

    # Statistics for each level
    for level_name, metrics in level_metrics.items():
        sample_count = len(metrics["gt_tools_list"])
        print(f"Number of Samples for {level_name}: {sample_count}")

    metrics_data = {
        "Proactive_Index_Accuracy": {
            "level1": calculate_accuracy(level_metrics["level1"]["pred_proactive_idx_list"], level_metrics["level1"]["gt_proactive_idx_list"], "Proactive Index")[0],
            "level2": calculate_accuracy(level_metrics["level2"]["pred_proactive_idx_list"], level_metrics["level2"]["gt_proactive_idx_list"], "Proactive Index")[0],
            "level3": calculate_accuracy(level_metrics["level3"]["pred_proactive_idx_list"], level_metrics["level3"]["gt_proactive_idx_list"], "Proactive Index")[0],
            "overall": calculate_accuracy(pred_proactive_idx_list, gt_proactive_idx_list, "Proactive Index")[0],
        },
        "Miss_Needed": {
            "level1": calculate_accuracy(level_metrics["level1"]["pred_proactive_idx_list"], level_metrics["level1"]["gt_proactive_idx_list"], "Proactive Index")[2],
            "level2": calculate_accuracy(level_metrics["level2"]["pred_proactive_idx_list"], level_metrics["level2"]["gt_proactive_idx_list"], "Proactive Index")[2],
            "level3": calculate_accuracy(level_metrics["level3"]["pred_proactive_idx_list"], level_metrics["level3"]["gt_proactive_idx_list"], "Proactive Index")[2],
            "overall": calculate_accuracy(pred_proactive_idx_list, gt_proactive_idx_list, "Proactive Index")[2],
        },
        "False_Detection": {
            "level1": calculate_accuracy(level_metrics["level1"]["pred_proactive_idx_list"], level_metrics["level1"]["gt_proactive_idx_list"], "Proactive Index")[3],
            "level2": calculate_accuracy(level_metrics["level2"]["pred_proactive_idx_list"], level_metrics["level2"]["gt_proactive_idx_list"], "Proactive Index")[3],
            "level3": calculate_accuracy(level_metrics["level3"]["pred_proactive_idx_list"], level_metrics["level3"]["gt_proactive_idx_list"], "Proactive Index")[3],
            "overall": calculate_accuracy(pred_proactive_idx_list, gt_proactive_idx_list, "Proactive Index")[3],
        },
        "Proactive_Score_Accuracy": {
            "level1": calculate_accuracy(level_metrics["level1"]["pred_proactive_score_list"], level_metrics["level1"]["gt_proactive_score_list"], "Proactive Index")[0],
            "level2": calculate_accuracy(level_metrics["level2"]["pred_proactive_score_list"], level_metrics["level2"]["gt_proactive_score_list"], "Proactive Index")[0],
            "level3": calculate_accuracy(level_metrics["level3"]["pred_proactive_score_list"], level_metrics["level3"]["gt_proactive_score_list"], "Proactive Index")[0],
            "overall": calculate_accuracy(pred_proactive_score_list, gt_proactive_score_list, "Proactive Index")[0],
        },
        "RMSE": {
            "level1": calculate_regression_metrics(
                list(map(float, level_metrics["level1"]["pred_proactive_score_list"])),
                list(map(float, level_metrics["level1"]["gt_proactive_score_list"])),
                "Proactive Score"
            )[1],
            "level2": calculate_regression_metrics(
                list(map(float, level_metrics["level2"]["pred_proactive_score_list"])),
                list(map(float, level_metrics["level2"]["gt_proactive_score_list"])),
                "Proactive Score"
            )[1],
            "level3": calculate_regression_metrics(
                list(map(float, level_metrics["level3"]["pred_proactive_score_list"])),
                list(map(float, level_metrics["level3"]["gt_proactive_score_list"])),
                "Proactive Score"
            )[1],
            "overall": calculate_regression_metrics(
                list(map(float, pred_proactive_score_list)),
                list(map(float, gt_proactive_score_list)),
                "Proactive Score"
            )[1],
        },
        "Precision_Tool_Names": {
            "level1": np.mean(level_metrics["level1"]["P_tool_names_list"]),
            "level2": np.mean(level_metrics["level2"]["P_tool_names_list"]),
            "level3": np.mean(level_metrics["level3"]["P_tool_names_list"]),
            "overall": np.mean(P_tool_names_list),
        },
        "Recall_Tool_Names": {
            "level1": np.mean(level_metrics["level1"]["R_tool_names_list"]),
            "level2": np.mean(level_metrics["level2"]["R_tool_names_list"]),
            "level3": np.mean(level_metrics["level3"]["R_tool_names_list"]),
            "overall": np.mean(R_tool_names_list),
        },
        "F1_Tool_Names": {
            "level1": np.mean(level_metrics["level1"]["F1_tool_names_list"]),
            "level2": np.mean(level_metrics["level2"]["F1_tool_names_list"]),
            "level3": np.mean(level_metrics["level3"]["F1_tool_names_list"]),
            "overall": np.mean(F1_tool_names_list),
        },
        "Argument_Accuracy": {
            "level1": 1 - (level_metrics["level1"]["arg_false"] / level_metrics["level1"]["arg_counts"]),
            "level2": 1 - (level_metrics["level2"]["arg_false"] / level_metrics["level2"]["arg_counts"]),
            "level3": 1 - (level_metrics["level3"]["arg_false"] / level_metrics["level3"]["arg_counts"]),
            "overall": 1 - (arg_false / arg_counts),
        }
    }

    # save scores to csv
    if methods == 'icl':
        csv_file = f'results/{dataset_name}/scores/metrics_{methods}_{model_base_icl}_fs_10_{args.personas}_{args.think}.csv'
    else:
        csv_file = f'results/{dataset_name}/scores/metrics_{methods}_{model_base_sft}_{args.personas}_{args.think}.csv'


    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Level1", "Level2", "Level3", "Overall"])
        for metric_name, values in metrics_data.items():
            writer.writerow([
                metric_name,
                round(values["level1"], 4),
                round(values["level2"], 4),
                round(values["level3"], 4),
                round(values["overall"], 4)
            ])

    print(f"Metrics summary saved to {csv_file}")

    print("Indices for each level:")
    for level, indices in level_indices.items():
        print(f"{level}: {indices}")
    print("Proactive examples: ", level_proactive)

    print("Predictions: ", pred_proactive_idx_list)
    print("GT: ", gt_proactive_idx_list)