# huggingface-cli login
# PYTHONPATH="${PYTHONPATH}:/rep/rabuhweidi/sft/SFTTraining-SemEval/src" python /rep/rabuhweidi/sft/SFTTraining-SemEval/src/blm/cli/inference.py \
#     --token hf_**** \
#     --model_name meta-llama/Llama-3.2-3B-Instruct


import argparse
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import os


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, required=True, help="Hugging Face API Token")
    # parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model")
    parser.add_argument('--model_name', type=str, required=True,
                        help="Pre-trained model name on Hugging Face (e.g., 'meta-llama/Llama-3.2-1B-Instruct')")
    args = parser.parse_args()
    hf_token = args.token  # Use the token provided in the argument

    if not hf_token:
        raise ValueError("Hugging Face token is required")

    # Define the chat template
    SYSTEMPROMPT = (
        """You are a strict traffic-classification model.

Rules:
• You will receive structured network traffic features.
• Select one numeric ID that matches the traffic type:
    0 = DDoS
    1 = DoS
    2 = Benign
    3 = Unknown
• Return ONLY the number.
"""
    )

    formatted_instructions = (
        """
        Classify the traffic based on the features below.
        
        Features: {flow_duration=-0.4120328702520254, Header_Length=-1.3813178440848826, Protocol Type=-0.9045590790028109, Duration=-0.1672118431027471, Rate=-0.2538017854346689, Drate=-0.0030510053339329, syn_flag_number=-0.5095636043371801, rst_flag_number=-0.3157239959867259, psh_flag_number=-0.3120838953733232, ack_flag_number=-0.3775236747416334, ece_flag_number=0.0, cwr_flag_number=0.0, syn_count=-0.5313075233141127, urg_count=-0.1514437905837197, rst_count=-0.3811637021800489, HTTP=0, HTTPS=0, DNS=0, Telnet=0.0, SMTP=0.0, SSH=-0.0067887836294242, IRC=0.0, TCP=0, UDP=0, DHCP=0.0, ARP=0, ICMP=1, IPv=1, Number=0.0032911819445537}
        
        LabelID:
        """
    )

    # Input data for the chat
    input_data = [
        {"content": SYSTEMPROMPT, "role": "system"},
        {"content": formatted_instructions, "role": "user"}
        # {"content": "", "role": "assistant"}  # Assistant's response will be generated  MK
    ]

    # Initialize the tokenizer and model
    model_path = args.model_name  # Get the model path from arguments
    tokenizer = AutoTokenizer.from_pretrained(model_path)  # Using the model name for tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)

    # MK
    # # Manually format the chat input for generation
    # chat_text = ""
    # for message in input_data:
    #     chat_text += f"{message['role']}: {message['content']}\n"

    prompt = tokenizer.apply_chat_template(input_data, tokenize=False, add_generation_prompt=True)
    generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer)  # , device=0)
    response = generator(prompt, max_length=512, num_return_sequences=1, return_full_text=True)
    print(f"\n\n{response}")
    inst_index = response[0]["generated_text"].find("[/INST]")
    relation = response[0]["generated_text"][(inst_index + len("[/INST]") + 1):].strip()
    print(f"relation_type: {relation}")


if __name__ == "__main__":
    main()
