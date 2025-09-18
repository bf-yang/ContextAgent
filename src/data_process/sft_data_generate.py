# Process the dataset into the CoT format for SFT
# <think> + <proactive predictions> + <toll_calling>
# Set a argument to determine whether include <think> and <personas> or not
import json
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='cab',help='cab, cab_lite, cab_ood')
    parser.add_argument("--mode", type=str, default='train',help='train, test')
    parser.add_argument("--think", type=str, default='w_t',help='w_t, wo_t')
    parser.add_argument("--personas", type=str, default='w_p',help='w_p, wo_p')
    args = parser.parse_args()

    dataset_name = args.dataset
    mode = args.mode
    PATH_BASE = f"data/{dataset_name}/"
    filename = f"{dataset_name}_{mode}.json"
    filename = f"{dataset_name}_{mode}.json"

    # Load dataset
    data_path = os.path.join(PATH_BASE, filename)
    with open(data_path, 'r') as f:
        dataset = json.load(f)
    print("Statistics of dataset:\n",dataset.keys())

    # Load prompt
    if args.think == "w_t":
        with open('prompt/prompt_sys.txt', 'r') as f:
            prompt_sys = f.read()
    elif args.think == "wo_t":
        with open('prompt/prompt_sys_wo_t.txt', 'r') as f:
            prompt_sys = f.read()


    # Build sft dataset
    dataset_processed = []
    for example in dataset:
        print("Example: ", example)
        data = dataset[example]
        instruction = prompt_sys # system prompt
        if dataset_name == "cab" or dataset_name == "cab_ood":
            contextual = data['Context information'] # contexual
        elif dataset_name == "cab_lite":
            contextual = data['Rawdata Context']
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}. Only 'cab' and 'cab_lite' are supported.")
        persona = data['Personas']       # persona
        think = data['Thoughts'] # think
        action = data['Action'] # actions
        tool_planning = data['Tool planning'] # tool planning

        input_text = "Sensory Context: \n" + contextual + "\n"
        output_text = ""

        # Personas
        if args.personas == "w_p":
            input_text += "Persona Context: \n" + ". ".join(persona) + "\n"

        # Thoughts
        if args.think == "w_t":
            output_text = f"<think>{data['Thoughts']}\n Actions:\n{action}\n <\\think>\n"
            # output_text = f"<think> {data['Thoughts']} <\\think>\n"
        
        # Proactive index and score
        output_text += " ## Proactive Predictions\n "
        output_text += f" \"Proactive index\": {data['Proactive index']}\n "
        output_text += f" \"Proactive score\": {data['Proactive score']}\n "

        # Tool Calling
        output_text += " ## Tool Calling:\n "
        output_text += f"{data['Tools']}\n"

        dataset_processed.append({
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        })

    # Save the json file for different experiments
    # sft.json
    if args.personas == "w_p" and args.think == "w_t":
        file_save = filename.replace(".json", f"_sft.json")
    # sft_wo_p.json
    if args.personas == "wo_p" and args.think == "w_t":
        file_save = filename.replace(".json", f"_sft_{args.personas}.json")
    # sft_wo_t.json
    if args.think == "wo_t" and args.personas == "w_p":   
        file_save = filename.replace(".json", f"_sft_{args.think}.json")
    # sft_wo_t_wo_p.json
    if args.think == "wo_t" and args.personas == "wo_p":   
        file_save = filename.replace(".json", f"_sft_{args.think}_{args.personas}.json")

    save_path = os.path.join(PATH_BASE, file_save)
    with open(save_path, 'w') as file:
        json.dump(dataset_processed, file, indent=2)