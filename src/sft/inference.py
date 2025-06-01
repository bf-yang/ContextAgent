# Evaluate SFT-based approaches
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
from utils import azure_inference,parse_proactive_agent_results
import argparse
import re
from tqdm import tqdm
import csv
import ast
import time

api_key = "4d2ff10a8c3d4d09883a4411832b6718" # Azure API key
client = AzureOpenAI(
    api_key = api_key,  
    api_version = "2023-05-15",
    azure_endpoint = "https://cuhk-aiot-gpt4.openai.azure.com/"
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_base", type=str, default='qwen7b',
                    help='base SFT model: qwen7b, llama8b, deepseek7b')
    parser.add_argument("--dataset", type=str, default='cab',help='cab, cab_lite, cab_ood')
    parser.add_argument("--think", type=str, default='w_t',help='w_t, wo_t')
    parser.add_argument("--personas", type=str, default='w_p',help='w_p, wo_p')
    args = parser.parse_args()

    # load the proactive agent's system prompt
    with open('prompt/prompt_sys.txt', 'r') as f:
        prompt_sys = f.read()

    # load sample data for evaluation
    dataset_name = args.dataset # dataset
    filepath = f'data/{dataset_name}/{dataset_name}_test.json'
    with open(filepath, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    print(dataset.keys())

    results_list = []
    for key in tqdm(dataset.keys()):
        sample = dataset[key]
        print("Sample ID:\n", key)
        print("="*50)

        # sensory context
        if dataset_name == 'cab' or dataset_name == 'cab_ood':
            contextual_info = sample['Context information']
        elif dataset_name == 'cab_lite':
            contextual_info = sample['Rawdata Context']
        # persona context
        personas_str = ".".join(sample['Personas'])
        print("\033[1;36mSensory Context:\033[0m\n", contextual_info)  
        print("="*50)
        print("\033[1;36mPersona Context:\033[0m\n", personas_str)  
        print("="*50)

        # proactive LLM agent inference
        client = OpenAI(
            api_key="{}".format(os.environ.get("API_KEY", "0")),
            base_url="http://localhost:{}/v1".format(os.environ.get("API_PORT", 8000)),
        )
        messages = []
        messages.append({"role": "system", "content": prompt_sys})
        if args.personas == 'w_p':
            messages.append({"role": "user", "content": "Sensory Context:\n"+contextual_info+"\nPersona Context:\n"+personas_str})
        else:
            messages.append({"role": "user", "content": "Sensory Context:\n"+contextual_info})

        result = client.chat.completions.create(messages=messages, model="test")
        result = result.choices[0].message.content
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
                        # Regenerate the tools
                        response = client.chat.completions.create(messages=messages, model="test")
                        result = response.choices[0].message.content
                        thoughts, proactive_idx, proactive_score, actions, tools = parse_proactive_agent_results(result)
                    else:
                        print("Max attempts reached. Unable to parse tools.")
                        json_tool = []
                        tools = tools + f" Max attempts reached. Unable to parse tools."

            results_tool = []
            # iterate over the tool calls
            if json_tool is not None: 
                for tool_call in json_tool:
                    print(50*"=")
                    if 'name' not in tool_call or 'parameters' not in tool_call:
                        results_tool.append({
                                    "tool_name": 'error',
                                    "tool_parameters": 'error',
                                    "results": 'error'
                                })
                    else:
                        print("Calling Function: ",tool_call['name'])
                        print("Function Params: ",tool_call['parameters'])
                        result_tool = process_function_call(tool_call)
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

        # Save results to json
        save_path = f'results/{dataset_name}/predictions/sft/pred_{args.model_base}_{args.personas}_{args.think}.json'
        with open(save_path, 'w', encoding='utf-8') as new_file:
            json.dump(dataset, new_file, ensure_ascii=False, indent=4)    