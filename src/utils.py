import re
import ollama
from ollama import chat
import transformers
from bert_score import score
from sklearn.metrics import mean_squared_error
import numpy as np

def azure_inference(client,model_id,messages,temperature,max_tokens):
    response = client.chat.completions.create(
        model=model_id,
        messages=messages, 
        temperature=temperature,
        max_tokens=max_tokens
        )
    response = response.choices[0].message.content
    return response

def ollama_inference(model_id,messages):
    response = ollama.chat(model=model_id, messages=messages,
                           options={"num_ctx":40960})
    return response['message']['content']


def parse_proactive_agent_results(results):
    '''Parse the proactive agent's results'''
    thoughts,proactive_idx,proactive_score, actions, tools = "None","None","None","None","None"

    # Parse <think> <\think> 
    thoughts_match = re.search(r'<think>(.*?)<\\think>', results, re.DOTALL)  
    if thoughts_match:  
        thoughts = thoughts_match.group(1)  
    
    # Parse "Proactive index" 
    proactive_idx_match = re.search(r'"Proactive index": (\w+)', results)  
    if proactive_idx_match:  
        proactive_idx = proactive_idx_match.group(1)  
    
    # Parse "Proactive score"  
    proactive_score_match = re.search(r'"Proactive score": (\d+)', results)  
    if proactive_score_match:  
        proactive_score = proactive_score_match.group(1)  

    action_match = re.search(r"## Action:\s*(.*?)\s*## Tool Calling:", results, re.DOTALL)  
    if action_match:  
        actions = action_match.group(1)

    # Parse "Tool Calling"  
    tool_match = re.search(r"## Tool Calling:\s*(\[.*?\]|None)", results, re.DOTALL)  
    if tool_match:  
        tools = tool_match.group(1)
    return thoughts, proactive_idx, proactive_score, actions, tools

def convert_sets_to_lists(obj):
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {key: convert_sets_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_sets_to_lists(element) for element in obj]
    else:
        return obj

# Evaluation metrics
def calculate_accuracy(pred_list, gt_list, label):
    if pred_list and gt_list:
        correct = sum(p == g for p, g in zip(pred_list, gt_list))
        accuracy = correct / len(gt_list)
        # print(f"Accuracy for {label}: {accuracy:.2%}")
        # 计算 miss-needed 和 false-detection
        miss_needed = sum(1 for p, g in zip(pred_list, gt_list) if g == "true" and p == "false") / len(gt_list)
        false_detection = sum(1 for p, g in zip(pred_list, gt_list) if g == "false" and p == "true") / len(gt_list)
        incorrect_indices = [i for i, (p, g) in enumerate(zip(pred_list, gt_list)) if p != g]
        return accuracy, incorrect_indices, miss_needed, false_detection

    else:
        print(f"No data available for calculating accuracy for {label}.")
        return []

def calculate_regression_metrics(pred_list, gt_list, label):
    if pred_list and gt_list:
        mse = mean_squared_error(gt_list, pred_list)
        rmse = np.sqrt(mse)
        return mse, rmse
    else:
        print(f"No data available for calculating regression metrics for {label}.")

def calculate_set_metrics(pred_set, gt_set, label):
    intersection = pred_set & gt_set
    precision = len(intersection) / len(pred_set) if pred_set else 0
    recall = len(intersection) / len(gt_set) if gt_set else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f"{label} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return precision, recall, f1