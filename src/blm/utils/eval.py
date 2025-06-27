# eval.py (Revised for Prompter __call__)
import argparse
import csv
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from huggingface_hub import login

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from datasets import load_from_disk
from blm.utils.prompter import Prompter
from blm.utils.log import log_message
import os
    
def extract_predicted_number_after_last_assistant(response): # I use it after train 
    if not response or 'generated_text' not in response[0]:
        return 3

    generated_text = response[0]['generated_text']
    match = re.search(r'LabelID:\s*(\d+)', generated_text)

    if match:
        return int(match.group(1)) # group(1) contains the captured digits
    else:
        return 3 #CICIoT_

def evaluate(model_path, test_data_path, hf_token, tokenizer_name,output_predictions_path,test_sample_limit,temperature,top_k,top_p): # Removed prompt paths
    """Evaluates the model on the provided test dataset using the Prompter."""
    log_message("INFO", "Evaluation start...")
    if hf_token:
        log_message("INFO", f"Logging into the Hugging Face Hub with token {hf_token[:10]}...")
        login(token=hf_token)

    log_message("INFO", f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    log_message("INFO", "Initializing the prompter...")
    prompter = Prompter(tokenizer)
    log_message("INFO", f"Prompter initialized successfully{prompter}.")
    log_message("INFO", f"Loading model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)
    model.eval()
    log_message("INFO", "Model loaded successfully in evaluation mode.")

    predictions = []
    ground_truth_labels = []
    evaluation_results = [] # To store individual results

    log_message("INFO", f"Loading test dataset from disk: {test_data_path}")
    try:
        test_dataset = load_from_disk(test_data_path)
        log_message("INFO", f"Test dataset loaded successfully with {test_dataset.num_rows} rows.")
    except FileNotFoundError:
        log_message("ERROR", f"Test dataset not found at: {test_data_path}")
        return
    except Exception as e:
        log_message("ERROR", f"Error loading test dataset: {e}")
        return
    
    log_message("INFO", "Mapping the prompter to the dataset...")
    test_dataset = test_dataset.map(prompter, batched=True)
    # log_message("INFO", "Prompter mapping to the dataset complete.")
    # log_message("INFO", f"test_dataset...\n{test_dataset}")
    # log_message("INFO", "Preparing prompts using the prompter...")
    promptCount=0
    for example in tqdm(test_dataset, desc="Evaluating"):
        log_message("INFO", f"example=========={example}")#rfa
        # log_message("INFO", f"test_dataset=========={test_dataset[0]}")#rfa
        # log_message("INFO", f"ground_truth_label==={example['messages'][2]['content']}")#rfa

        prompt = example['prompt']
        ground_truth_label = int(example['messages'][2]['content'])#rfa
        sentence= example['messages'][1]['content']#rfa
        ground_truth_labels.append(ground_truth_label)
        log_message("INFO", f"\n-------------------------------------------------------------------------")
        log_message("INFO", f"Prompt:({promptCount}) ")
        log_message("INFO", f"-------------")
        log_message("INFO", f"Look Here ... prompt is:{prompt}=== End Look")
        # inputs = tokenizer(prompt, return_tensors="pt").to(model.device) --- back
        # log_message("INFO", f"------------inputs-{inputs}")

        # Create the text generation pipeline
        generator = pipeline(
            task="text-generation", 
            model=model, 
            tokenizer=tokenizer
            # device_map="auto"
            )
        # Generate the response
        generated_text = generator(
            sentence,
            #max_length=1024, 
            num_return_sequences=1, 
            return_full_text=True,
            max_new_tokens=1,
            temperature=temperature ,
            do_sample=False
            # top_k=top_k,
            # top_p=top_p
            )
        
        
        # #or-base
        # generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer) #, device=0)
        # response = generator(prompt, max_length=1024, num_return_sequences=1, return_full_text=True)
        
        print(f"RRRRRRRRRRRRRRRRRR==========generated_text:-{generated_text}===")
             
        
        # prompt = tokenizer.apply_chat_template(inputs, tokenize=False, add_generation_prompt=True)
        # with torch.no_grad():
        #     outputs = model.generate(**inputs, max_new_tokens=1, do_sample=False) ---back
        #     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)  ------back
            
            # log_message("INFO", f"==========generated_text:-{generated_text}===")

       
        predicted_label=extract_predicted_number_after_last_assistant(generated_text)#rfa
        log_message("INFO", f"Generated response: ")
        log_message("INFO", f"-------------")
        log_message("INFO", f" Look Here ... generated response is:{generated_text}=== End Look")#rfa
        # log_message("NFO", f" Look Here ... predicted_label is:{predicted_label}=== End Look")#rfa

        predictions.append(predicted_label)
        log_message("INFO", f"Ground Truth Label: {ground_truth_label}, Predicted Label: {predicted_label}, 'sentence': {sentence}, 'generated_text': {generated_text}") #rfa
        log_message("INFO", f"--------------------------------------------------------")
        evaluation_results.append({'actual_value': ground_truth_label, 'predicted_value': predicted_label, 'sentence': sentence, 'generated_text': generated_text})#rfa
        promptCount=promptCount+1
        if promptCount >= test_sample_limit:
            break
    # Save individual predictions if an output path is provided
    output_directory = os.path.dirname(output_predictions_path)
    os.makedirs(output_directory, exist_ok=True)
    if output_predictions_path:
        log_message("INFO", f"Saving individual predictions to: {output_predictions_path}")
        with open(output_predictions_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['actual_value', 'predicted_value', 'sentence', 'generated_text']#rfa
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(evaluation_results)
        log_message("INFO", "Individual predictions saved successfully.")

    # Calculate metrics
    valid_predictions = [p for p in predictions if p is not None]
    valid_ground_truth = [g for i, g in enumerate(ground_truth_labels) if predictions[i] is not None]


    if not valid_ground_truth:
        log_message("WARNING", "No valid predictions found for metric calculation.")
        accuracy = 0.0
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    else:
        accuracy = accuracy_score(valid_ground_truth, valid_predictions)
        precision = precision_score(valid_ground_truth, valid_predictions, average="macro", zero_division=0)#rfa
        recall = recall_score(valid_ground_truth, valid_predictions, average="macro", zero_division=0)#rfa
        f1 = f1_score(valid_ground_truth, valid_predictions, average="macro", zero_division=0)#rfa
        log_message("INFO", f"Evaluation Accuracy: {accuracy * 100:.2f}%")
        log_message("INFO", f"Evaluation Precision: {precision * 100:.2f}%")
        log_message("INFO", f"Evaluation Recall: {recall * 100:.2f}%")
        log_message("INFO", f"Evaluation F1 Score: {f1 * 100:.2f}%")
        
log_message("INFO", f"Evaluation end")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate a language model for common sense reasoning")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved merged model directory")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the HuggingFace dataset directory for test data")
    parser.add_argument("--hf_token", type=str, default=None, help="Huggingface token (optional)")
    parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Tokenizer name or path")
    parser.add_argument("--output_predictions_path", type=str, default=None, help="Path to save individual predictions (optional)")
    parser.add_argument("--test_sample_limit", type=int, default=10, help="no of test sample used in evaluation (optional)")

    parser.add_argument("--temperature", type=int, default=0, help="tempreture")
    parser.add_argument("--top_k", type=int, default=1, help="top_k")
    parser.add_argument("--top_p", type=int, default=1, help="top_p")

    args = parser.parse_args()
    evaluate(args.model_path, args.test_data_path, args.hf_token, args.tokenizer_name, args.output_predictions_path,args.test_sample_limit,args.temperature,args.top_k,args.top_p)
