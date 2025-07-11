## Benchmarking AI models for IoT Network Intrusion Detection: An Analysis of Performance and Time Efficiency

COMP9340 – Computer Security Course - 2025

- Lecture: Prof. [Mohaiesn Daivad](https://www.cs.ucf.edu/~mohaisen/)
- Students: [Ruwa AbuHweidi](https://github.com/RuwaYafa/IoT-IDS-LLM) and [Monther Salahat](https://github.com/msalahat2015/IoT-IDS)

---
## ▶️ Methodology Pipeline
![Methodology Pipeline](IoT-Methodology.png)
---
* Our Project has two evaluation parts: Deep Learning and Machine Learning code are available on this repository [Monther Salahat](https://github.com/msalahat2015/IoT-IDS), and LLM code [Ruwa AbuHweidi](https://github.com/RuwaYafa/IoT-IDS-LLM)
* All Results are uploaded in repository [Dropbox](https://www.dropbox.com/scl/fo/7y8a8j7tko3da90sr6mco/AHpHYC95o7b65hER_MAKjKs?rlkey=2m6asv519j7w6li4d2c45ecbo&dl=0):
  * [Logs](https://www.dropbox.com/scl/fo/62zortvc8kuwud8ptn1n0/AEUulu3p5iGV0338tpijExU?rlkey=oaj70ynnmv2dtruwp5j4obs9v&dl=0).
  * [Saved models](https://drive.google.com/drive/folders/14io4lIMozrjQo1An5drDHKUwH6-ukY3X?usp=sharing) (Machine and Deep Learning).
  * [Checkpoints](https://www.dropbox.com/scl/fo/yg306y5df9y5eyjp9mz72/AKbDfF1MVBljfVSORMXKsC4?rlkey=wgv7mnhscnbvij32v12qd8lgo&dl=0) for trained LLM models.
---
## ▶️ Reproducability
---
### Project structure - LLM Part
<pre><code>
.
├── .venv/
├── input_folder/
├── eval/
│   ├── data-00000-of-00001.arrow
│   ├── dataset_info.json
│   └── state.json
├── train/
│   └── dataset_dict.json
├── merge_lora/ #SFT model
├── model/ #checkpoints 
├── output_folder/
├── src/
│   ├── blm/
│   │   ├── cli/
│   │   │   ├── inference.py
│   │   │   ├── merge_lora.py
│   │   │   ├── process.py
│   │   │   └── train.py
│   │   ├── config/
│   │   │   ├── deepspeed_zero2.json
│   │   │   └── deepspeed_zero3.json
│   │   ├── prompts/
│   │   │   ├── Prompt_template.png
│   │   │   ├── system_prompt.txt
│   │   │   └── user_prompt.txt
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   ├── calc-maxlength.py
│   │   │   ├── eval.py
│   │   │   ├── helpers.py
│   │   │   ├── log.py
│   │   │   ├── peft.py
│   │   │   ├── prompter.py
│   │   │   ├── read.py
│   │   │   ├── save_to_hfhub.py
│   │   │   └── train.py
│   │   ├──__init__.py
│   │   ├──environment.yml
│   │   ├──requirements.txt
│   │   ├──requirements-server.txt
│   │   └──visualization.txt
├── IoT-Methodology.png
├── LICENSE
├── pyrightconfig.json
└── README.md
</code></pre>

To reproduce our work you can follow the next steps after change the paths depends on your machine.
---
### Initial Step to prepare your environment: 
  
  - git clone https://github.com/RuwaYafa/IoT-IDS-LLM
  - conda create -n <env name> python=3.12 
  - conda activate <env name>
  - pip install -r requirements.txt

---
## ▶️ Data processing: the same for machine learning and Deep learning parts
### pre-process:
    export PYTHONPATH="${PYTHONPATH}:/rep/rabuhweidi/sft/SFTTraining-SemEval/src"
    export PYTHONPATH="${PYTHONPATH}:/rep/rabuhweidi/LLMTraining/src"
    python /rep/rabuhweidi/sft/SFTTraining-SemEval/src/blm/cli/process.py \
        --output_path /rep/rabuhweidi/sft/SFTTraining-SemEval/input_folder \
        --n 
---
## ▶️ Training
### SFT - The Training Process
    CUDA_LAUNCH_BLOCKING=1 PYTHONPATH="$PYTHONPATH:/rep/rabuhweidi/LLMTraining/src" deepspeed --num_gpus=2 /rep/rabuhweidi/LLMTraining/src/blm/cli/train.py \
        --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
        --quantize True \
        --token hf_**** \
        --data_path /rep/rabuhweidi/sft/SFTTraining-SemEval/input_folder \
        --max_seq_length 1024 \
        --num_train_epochs 1 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --gradient_checkpointing True \
        --learning_rate 0.0002 \
        --weight_decay 0.0 \
        --bf16 True \
        --tf32 False \
        --output_dir /rep/rabuhweidi/sft/SFTTraining-SemEval/output_folder/CICIoT_2023/mistralai-CICIoT_2023 \
        --logging_strategy steps \
        --logging_steps 40 \
        --eval_strategy steps \
        --eval_accumulation_steps 200 \
        --save_steps 200 \
        --deepspeed /rep/rabuhweidi/sft/SFTTraining-SemEval/src/blm/config/deepspeed_zero3.json \
        --lora_r 64 \
        --lora_alpha 16 \
        --lora_dropout 0.1
---
## ▶️ Evaluation
### evaluation-baseline: 
    PYTHONPATH="${PYTHONPATH}:/rep/rabuhweidi/sft/SFTTraining-SemEval/src" python /rep/rabuhweidi/sft/SFTTraining-SemEval/src/blm/utils/eval.py \
        --model_path mistralai/Mistral-7B-Instruct-v0.2 \
        --test_data_path /rep/rabuhweidi/sft/SFTTraining-SemEval/input_folder/eval \
        --token hf_**** \        
        --tokenizer_name mistralai/Mistral-7B-Instruct-v0.2 \
        --output_predictions_path /rep/rabuhweidi/sft/SFTTraining-SemEval/output_folder/CICIoT_2023/test.csv \
        --test_sample_limit 2240 \
        #--temperature  0 \
        #--top_p 1 \
        #--top_k 1

---
## ▶️ Model 
### Marge Lora: 
    CUDA_LAUNCH_BLOCKING=1 PYTHONPATH="${PYTHONPATH}:/rep/rabuhweidi/LLMTraining/src" python "/rep/rabuhweidi/LLMTraining/src/blm/cli/merge_lora.py" \
        --lora_path "/rep/rabuhweidi/sft/SFTTraining-SemEval/output_folder/CICIoT_2023/mistralai-CICIoT_2023/checkpoint-1200" \
        --merged_path "/rep/rabuhweidi/sft/SFTTraining-SemEval/model/try7_Mistral-7B-Instruct-v0.2-CICIoT_2023" \
        --hf_token "hf_****"

---
## ▶️ Cont. Evaluation
### evaluation-after SFT: 
    PYTHONPATH="${PYTHONPATH}:/rep/rabuhweidi/sft/SFTTraining-SemEval/src" python /rep/rabuhweidi/sft/SFTTraining-SemEval/src/blm/utils/eval.py \
        --model_path /rep/rabuhweidi/sft/SFTTraining-SemEval/model/try7_Mistral-7B-Instruct-v0.2-CICIoT_2023 \
        --test_data_path /rep/rabuhweidi/sft/SFTTraining-SemEval/input_folder/eval \
        --hf_token hf_**** \
        --tokenizer_name /rep/rabuhweidi/sft/SFTTraining-SemEval/model/try7_Mistral-7B-Instruct-v0.2-CICIoT_2023 \
        --output_predictions_path /rep/rabuhweidi/sft/SFTTraining-SemEval/output_folder/CICIoT_2023/predict_log_Mistral_CICoT_SFT \
        --test_sample_limit 2240 \
        #--temperature  0 \
        #--top_p 1 \
        #--top_k 1
---
## ▶️ Deploy
### save on huggingface:
    CUDA_LAUNCH_BLOCKING=1 PYTHONPATH="${PYTHONPATH}:/rep/rabuhweidi/sft/SFTTraining-SemEval/src" python /rep/rabuhweidi/sft/SFTTraining-SemEval/src/blm/utils/save_to_hfhub.py \
        --model_path "/rep/rabuhweidi/sft/SFTTraining-SemEval/output_folder/CICIoT_2023/model_try7_Mistral-7B-Instruct-v0.2-CICIoT_2023/mistralai" \
        --repo_name "RuwaYafa/Mistral-7B-Instruct-v0.2-CICIoT_2023" \
        --hf_token "hf_****" \
        --your_name "Tarqeem" \
        #--private  # Remove this for public repo
---
## ▶️ Results
![t-SNE projection of algorithm performance vectors, illustrating the clustering of DL,ML, and LLM algorithms based
on their accuracy, precision, recall, F1-score, and log-scaled training time, prediction time, and model size](output_folder/tsne_algorithms.png)
### Our Related Links:

> SFT Models on Hugging face

[Mistral-7B SFT model](https://huggingface.co/RuwaYafa/Mistral-7B-Instruct-v0.2-CICIoT_2023)
    
[Llama-1B SFT model](https://huggingface.co/RuwaYafa/Llama-3.2-1B-Instruct-CICIoT_2023)
    
[Llama-3B SFT model](https://huggingface.co/RuwaYafa/Llama-3.2-3B-Instruct-CICIoT_2023)
    
> Dataset

[Dataset: Tarqeem/ICICoT-2023](https://huggingface.co/datasets/Tarqeem/ICICoT-2023)

---

### ▶️ Used Prompt:

![Prompt Template](src/blm/prompts/Prompt_template.png)

    System prompt:
    You are a strict traffic-classification model.
    
    Rules:
    • You will receive structured network traffic features.
    • Select one numeric ID that matches the traffic type:
        0 = DDoS
        1 = DoS
        2 = Benign
        3 = Unknown
    • Return ONLY the number.
---
    User prompt:
    Classify the traffic based on the features below.
    
    Features: {sentence}
    
    LabelID:
---
#### Acknowledgement
This project thanks the open repository of [@mohammedkhalilia](https://github.com/mohammedkhalilia/LLMTraining/tree/main), which provides a starting point for developing our code.
