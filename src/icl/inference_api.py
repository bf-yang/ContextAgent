# Evaluate ICL-based approaches via API
# ICL with 10-shots
# Argments:
#   -- zs
#   -- think
#   -- personas
#   -- n_fewshot

# Baselines
#   -- Zero-Shot:     --n_fewshot=0, --think='wo_t', --personas='wo_p'
#   -- Context-only:  --n_fewshot=10, --think='wo_t', --personas='wo_p'
#   -- CoT:   --n_fewshot=10, --think='w_t', --personas='wo_p'
#   -- ICL-P: --n_fewshot=10, --think='wo_t', --personas='w_p'
import json
import os
from typing import Sequence
from openai import OpenAI
from transformers.utils.versions import require_version
require_version("openai>=1.5.0", "To fix: pip install openai>=1.5.0")
import json
import sys 
sys.path.append("/home/bufang/ContextAgent/src") 
from functions import *
import ollama
from openai import AzureOpenAI
from utils import azure_inference, ollama_inference, \
    parse_proactive_agent_results,convert_sets_to_lists
import argparse
import re
from tqdm import tqdm
import ast
import random
random.seed(42)
api_key = "4d2ff10a8c3d4d09883a4411832b6718" # Azure API key
client = AzureOpenAI(
    api_key = api_key,  
    api_version = "2023-05-15",
    azure_endpoint = "https://cuhk-aiot-gpt4.openai.azure.com/"
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_base", type=str, default='gpt-35-turbo',
                        help='gpt-35-turbo" "gpt-4o-2" "gpt-4o-mini-2')    
    parser.add_argument("--dataset", type=str, default='cab',help='cab, cab_lite, cab_ood')
    parser.add_argument("--zs", type=str, default='false',help='false, true')
    parser.add_argument("--personas", type=str, default='w_p',help='w_p, wo_p')
    parser.add_argument("--think", type=str, default='w_t',help='w_t, wo_t')
    parser.add_argument("--n_fewshot", type=int, default='10', help='Number of few-shot examples')
    args = parser.parse_args()

    dataset_name = args.dataset # dataset

    # load the proactive agent's system prompt
    if args.zs == 'true':
        with open('prompt/baselines/icl_zs.txt', 'r') as f:
            prompt_sys = f.read()
    else:
        with open('prompt/baselines/icl_fs.txt', 'r') as f:
            prompt_sys = f.read()

    # load seed dataset
    path = f'data/{dataset_name}/{dataset_name}_train.json'
    with open(path, 'r') as f:
        dataset = json.load(f)
    keys = list(dataset.keys())
    
    # Random select n samples for ICL
    sample_keys = random.sample(keys, args.n_fewshot)
    print("Selected samples: ", sample_keys)
    samples = {key: dataset[key] for key in sample_keys}

    # Delete "Response" in the demonstrations. Keep the same as SFT.
    for key in samples:
        if "Response" in samples[key]:
            del samples[key]["Response"]
        if "Tool planning" in samples[key]:
                del samples[key]["Tool planning"]
        if "Action" in samples[key]:
            del samples[key]["Action"]
    if args.think == 'wo_t': # without thoughts in ICL
        for key in samples:
            if "Thoughts" in samples[key]:
                del samples[key]["Thoughts"]
    if args.personas == 'wo_p': # without personas in ICL
        for key in samples:
            if "Personas" in samples[key]:
                del samples[key]["Personas"]
    samples_str = json.dumps(samples, ensure_ascii=False, indent=4)

    # Add examples to prompt
    if args.zs == 'true':
        pass
    else:
        prompt_sys = prompt_sys.replace("{Examples}", samples_str)
    print("Prompt:\n"+prompt_sys)

    # load sample data for evaluation
    filepath = f'data/{dataset_name}/{dataset_name}_test.json'
    with open(filepath, 'r', encoding='utf-8') as file:
        dataset = json.load(file)

    results_list = []
    for key in tqdm(dataset.keys()):
        sample = dataset[key]
        print("Sample ID:\n", key)
        print("="*50)

        # sensory context
        contextual_info = sample['Context information']
        # persona context
        personas_str = ".".join(sample['Personas'])
        print("\033[1;36mSensory Context:\033[0m\n", contextual_info)  
        print("="*50)

        # proactive LLM agent inference
        messages = []
        messages.append({"role": "system", "content": prompt_sys})
        if args.personas == 'w_p':
            messages.append({"role": "user", "content": "Sensory Context:\n"+contextual_info+"\nPersona Context:\n"+personas_str})
        else:
            messages.append({"role": "user", "content": contextual_info})

        # LLM inference via API
        try:
            result = azure_inference(client, args.model_base, messages, temperature=0.7, max_tokens=4096)
        except Exception as e:
            result = "None"
        print("\033[1;36mProactive LLM Agent Predictions:\033[0m\n", result)  
        print("="*50)

        # Parse the proactive agent's results
        thoughts, proactive_idx, proactive_score, actions, tools = parse_proactive_agent_results(result)
        print("\033[1;36mThoughts:\033[0;36m\n", thoughts, "\033[0m")  
        print("\033[1;34mProactive Index:\033[0;34m\n", proactive_idx, "\033[0m")  
        print("\033[1;34mProactive Score:\033[0;34m\n", proactive_score, "\033[0m")  
        print("\033[1;37mActions:\033[0;37m\n", actions, "\033[0m")  
        print("\033[1;35mTools:\033[0;35m\n", tools, "\033[0m") 

        # tool calling
        if tools != 'None':
            # Check if tools is a valid JSON string
            max_attempts = 1
            attempt = 0
            json_tool = None
            while attempt < max_attempts:
                try:
                    json_tool = ast.literal_eval(tools)
                    break
                except (ValueError, SyntaxError) as e:
                    print(f"Attempt {attempt + 1}: Error parsing tools with ast.literal_eval: {e}")
                    attempt += 1
                    if attempt < max_attempts:
                        # Retry if there is an error for tool calling
                        print("Retrying tool calling...")
                        result = ollama_inference(args.model_base,messages)
                        thoughts, proactive_idx, proactive_score, actions, tools = parse_proactive_agent_results(result)
                    else:
                        print("Max attempts reached. Unable to parse tools.")
                        json_tool = []
                        tools = tools + f" Max attempts reached. Unable to parse tools."

            # Tool Execution
            results_tool = []
            if json_tool is None:
                print("Unable to parse tools.")
            else:
                # iteratively call the tools
                for tool_call in json_tool:
                    if not isinstance(tool_call, dict) or'name' not in tool_call or 'parameters' not in tool_call:
                        print("Invalid tool call format:", tool_call)
                        results_tool.append({
                            "tool_name": 'error',
                            "tool_parameters": 'error',
                            "results": 'error'
                        })
                    else:
                        print(50*"=")
                        print("Calling Function: ",tool_call['name'])
                        print("Function Params: ",tool_call['parameters'])
                        result_tool = process_function_call(tool_call) # execute the tool
                        print("Function Results: ",result_tool)
                        results_tool.append({
                                "tool_name": tool_call['name'],
                                "tool_parameters": tool_call['parameters'],
                                "results": result_tool
                            })
                print(50*"=")
                print("\033[1;36mTool Results:\033[0m\n", results_tool)  

            # LLM reasoning on contextual and function results 
            with open('prompt/prompt_summarize.txt', 'r', encoding='utf-8') as file:
                prompt_summarize = file.read()
            if args.personas == 'wo_p':
                content = "# Sensory Contexts:\n"+contextual_info+"\n\n# Thoughts:\n"+thoughts+"\n\n# Tool results:\n"+str(results_tool)
            else:
                content = "# Sensory Contexts:\n"+contextual_info+"\n\n# Persona Contexts:\n"+personas_str+"\n\n# Thoughts:\n"+thoughts+"\n\n# Tool results:\n"+str(results_tool)              
            massages = [
                {'role': 'system','content': prompt_summarize},
                {'role': 'user','content': content}
                ]

            print(50*"=")
            print(massages)
            print(50*"=")
            # LLM reasoning
            # response = ollama.chat(model='qwen2.5', messages=massages)
            # response = response['message']['content']
            # response = azure_inference(client, args.model_base, massages, temperature=0.7, max_tokens=100)
            # print(response)
            response = 'outputs.'
            print("\033[1;36mResponse:\033[0m\n", response)  

            sample['predictions'] = {
                'thoughts': thoughts,
                'proactive_idx': proactive_idx,
                'proactive_score': proactive_score,
                'actions': actions,
                'tools': tools,
                'tools_results': results_tool,
                'response': response
            }
        else:
            sample['predictions'] = {
                'thoughts': thoughts,
                'proactive_idx': proactive_idx,
                'proactive_score': proactive_score,
                'actions': actions,
                'tools': tools,
                'tools_results': 'None',
                'response': 'None'
            }

        dataset = convert_sets_to_lists(dataset)
        # Save results to json
        if args.zs == 'true':
            save_path = f'results/{dataset_name}/predictions/icl/pred_{args.model_base}_zs.json'
        else:
            save_path = f'results/{dataset_name}/predictions/icl/pred_{args.model_base}_fs_{str(args.n_fewshot)}_{args.personas}_{args.think}.json'
        with open(save_path, 'w', encoding='utf-8') as new_file:
            json.dump(dataset, new_file, ensure_ascii=False, indent=4)    