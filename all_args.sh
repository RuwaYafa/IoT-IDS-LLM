#Ruwa_AbuHweidi&MontherSalahat@2025

Data processing: the same for machine learning and Deep learning parts

pre-process:
export PYTHONPATH="${PYTHONPATH}:/rep/rabuhweidi/sft/SFTTraining-SemEval/src" #or
export PYTHONPATH="${PYTHONPATH}:/rep/rabuhweidi/LLMTraining/src"

[------------------------------------------------------------------]

python /rep/rabuhweidi/sft/SFTTraining-SemEval/src/blm/cli/process.py \
	--output_path /rep/rabuhweidi/sft/SFTTraining-SemEval/input_folder \
	--n 

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
meta-llama/Llama-3.2-1B-Instruct
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

SFT - The training Process
CUDA_LAUNCH_BLOCKING=1 PYTHONPATH="$PYTHONPATH:/rep/rabuhweidi/LLMTraining/src" deepspeed --num_gpus=2 /rep/rabuhweidi/LLMTraining/src/blm/cli/train.py \
    --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
    --quantize True \
    --token hf_***** \
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

[------------------------------------------------------------------]
evaluation-baseline: 
PYTHONPATH="${PYTHONPATH}:/rep/rabuhweidi/sft/SFTTraining-SemEval/src" python /rep/rabuhweidi/sft/SFTTraining-SemEval/src/blm/utils/eval.py \
    --model_path mistralai/Mistral-7B-Instruct-v0.2 \
    --test_data_path /rep/rabuhweidi/sft/SFTTraining-SemEval/input_folder/eval \
    --hf_token hf_***** \
    --tokenizer_name mistralai/Mistral-7B-Instruct-v0.2 \
    --output_predictions_path /rep/rabuhweidi/sft/SFTTraining-SemEval/output_folder/CICIoT_2023/test.csv \
    --test_sample_limit 2240 \
    --temperature  0 \
    --top_p 1 \
    --top_k 1

[------------------------------------------------------------------]

marge lora: 
CUDA_LAUNCH_BLOCKING=1 PYTHONPATH="${PYTHONPATH}:/rep/rabuhweidi/LLMTraining/src" python "/rep/rabuhweidi/LLMTraining/src/blm/cli/merge_lora.py" \
    --lora_path "/rep/rabuhweidi/sft/SFTTraining-SemEval/output_folder/CICIoT_2023/mistralai-CICIoT_2023/checkpoint-1200" \
    --merged_path "/rep/rabuhweidi/sft/SFTTraining-SemEval/model/try7_Mistral-7B-Instruct-v0.2-CICIoT_2023" \
    --hf_token "hf_*****"

[------------------------------------------------------------------]
evaluation-after SFT: 
PYTHONPATH="${PYTHONPATH}:/rep/rabuhweidi/sft/SFTTraining-SemEval/src" python /rep/rabuhweidi/sft/SFTTraining-SemEval/src/blm/utils/eval.py \
    --model_path /rep/rabuhweidi/sft/SFTTraining-SemEval/model/try7_Mistral-7B-Instruct-v0.2-CICIoT_2023 \
    --test_data_path /rep/rabuhweidi/sft/SFTTraining-SemEval/input_folder/eval \
    --hf_token hf_***** \
    --tokenizer_name /rep/rabuhweidi/sft/SFTTraining-SemEval/model/try7_Mistral-7B-Instruct-v0.2-CICIoT_2023 \
    --output_predictions_path /rep/rabuhweidi/sft/SFTTraining-SemEval/output_folder/CICIoT_2023/predict_log_Mistral_CICoT_SFT.csv \
    --test_sample_limit 2240 \
    --temperature  0 \
    --top_p 1 \
    --top_k 1

save on huggingface:// not work in server
CUDA_LAUNCH_BLOCKING=1 PYTHONPATH="${PYTHONPATH}:/rep/rabuhweidi/sft/SFTTraining-SemEval/src" python /rep/rabuhweidi/sft/SFTTraining-SemEval/src/blm/utils/save_to_hfhub.py \
    --model_path "/rep/rabuhweidi/sft/SFTTraining-SemEval/output_folder/CICIoT_2023/model_try7_Mistral-7B-Instruct-v0.2-CICIoT_2023/mistralai" \
    --repo_name "RuwaYafa/Mistral-7B-Instruct-v0.2-CICIoT_2023" \
    --hf_token "hf_*****" \
    --your_name "Tarqeem" \
#    --private  # Remove this for public repo

Colab Test: 

Repeat in server:

CUDA_LAUNCH_BLOCKING=1 PYTHONPATH="$PYTHONPATH:/rep/rabuhweidi/LLMTraining/src" deepspeed --num_gpus=2 /rep/rabuhweidi/LLMTraining/src/blm/cli/train.py \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --quantize True \
    --token hf_***** \
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
    --output_dir /rep/rabuhweidi/sft/SFTTraining-SemEval/output_folder/CICIoT_2023/Llama_1B-CICIoT_2023 \
    --logging_strategy steps \
    --logging_steps 40 \
    --eval_strategy steps \
    --eval_accumulation_steps 200 \
    --save_steps 200 \
    --deepspeed /rep/rabuhweidi/sft/SFTTraining-SemEval/src/blm/config/deepspeed_zero3.json \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.1

marge lora: 
CUDA_LAUNCH_BLOCKING=1 PYTHONPATH="${PYTHONPATH}:/rep/rabuhweidi/LLMTraining/src" python "/rep/rabuhweidi/LLMTraining/src/blm/cli/merge_lora.py" \
    --lora_path "/rep/rabuhweidi/sft/SFTTraining-SemEval/output_folder/CICIoT_2023/Llama_1B-CICIoT_2023/checkpoint-1260" \
    --merged_path "/rep/rabuhweidi/sft/SFTTraining-SemEval/output_folder/CICIoT_2023/model_Llama1B_CICIoT_2023" \
    --hf_token "hf_*****"

evaluation-baseline:
PYTHONPATH="${PYTHONPATH}:/rep/rabuhweidi/sft/SFTTraining-SemEval/src" python /rep/rabuhweidi/sft/SFTTraining-SemEval/src/blm/utils/eval.py \
    --model_path meta-llama/Llama-3.2-1B-Instruct \
    --test_data_path /rep/rabuhweidi/sft/SFTTraining-SemEval/input_folder/eval \
    --hf_token hf_***** \
    --tokenizer_name meta-llama/Llama-3.2-1B-Instruct \
    --output_predictions_path /rep/rabuhweidi/sft/SFTTraining-SemEval/output_folder/CICIoT_2023/logs_Results/predict_log_Llama1B_CICoT_SFT.csv \
    --test_sample_limit 2240 \
    --temperature  0 \
    --top_p 1 \
    --top_k 1

evaluation-after SFT: 
PYTHONPATH="${PYTHONPATH}:/rep/rabuhweidi/sft/SFTTraining-SemEval/src" python /rep/rabuhweidi/sft/SFTTraining-SemEval/src/blm/utils/eval.py \
    --model_path /rep/rabuhweidi/sft/SFTTraining-SemEval/output_folder/CICIoT_2023/model_Llama1B_CICIoT_2023 \
    --test_data_path /rep/rabuhweidi/sft/SFTTraining-SemEval/input_folder/eval \
    --hf_token hf_***** \
    --tokenizer_name /rep/rabuhweidi/sft/SFTTraining-SemEval/output_folder/CICIoT_2023/model_Llama1B_CICIoT_2023 \
    --output_predictions_path /rep/rabuhweidi/sft/SFTTraining-SemEval/output_folder/CICIoT_2023/logs_Results/predict_log_Llama1B_CICoT_baseline.csv \
    --test_sample_limit 2240 \
    --temperature  0 \
    --top_p 1 \
    --top_k 1

save on huggingface:// not work in server
CUDA_LAUNCH_BLOCKING=1 PYTHONPATH="${PYTHONPATH}:/rep/rabuhweidi/sft/SFTTraining-SemEval/src" python /rep/rabuhweidi/sft/SFTTraining-SemEval/src/blm/utils/save_to_hfhub.py \
    --model_path "/rep/rabuhweidi/sft/CICIoT_2023/Llama1B_CICIoT_2023/model_Llama1B_CICIoT_2023" \
    --repo_name "RuwaYafa/Llama-3.2-1B-Instruct-CICIoT_2023" \
    --hf_token "hf_*****" \
    --your_name "Tarqeem" \
#    --private  # Remove this for public repo

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
meta-llama/Llama-3.2-3B-Instruct
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

Colab Test: 

Repeat in server:

CUDA_LAUNCH_BLOCKING=1 PYTHONPATH="$PYTHONPATH:/rep/rabuhweidi/LLMTraining/src" deepspeed --num_gpus=2 /rep/rabuhweidi/LLMTraining/src/blm/cli/train.py \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --quantize True \
    --token hf_***** \
    --data_path /rep/rabuhweidi/sft/CICIoT_2023/data_CICIoT \
    --max_seq_length 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
    --learning_rate 0.0002 \
    --weight_decay 0.0 \
    --bf16 True \
    --tf32 False \
    --output_dir /rep/rabuhweidi/sft/SFTTraining-SemEval/output_folder/CICIoT_2023/Llama_3B-CICIoT_2023 \
    --logging_steps 40 \
    --eval_strategy steps \
    --eval_accumulation_steps 200 \
    --save_steps 200 \
    --deepspeed /rep/rabuhweidi/sft/SFTTraining-SemEval/src/blm/config/deepspeed_zero3.json \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.1

marge lora: 
CUDA_LAUNCH_BLOCKING=1 PYTHONPATH="${PYTHONPATH}:/rep/rabuhweidi/LLMTraining/src" python "/rep/rabuhweidi/LLMTraining/src/blm/cli/merge_lora.py" \
    --lora_path "/rep/rabuhweidi/sft/SFTTraining-SemEval/output_folder/CICIoT_2023/Llama_3B-CICIoT_2023/checkpoint-994" \
    --merged_path "/rep/rabuhweidi/sft/SFTTraining-SemEval/output_folder/CICIoT_2023/model_Llama3B_CICIoT_2023" \
    --hf_token "hf_*****"

evaluation-baseline:
PYTHONPATH="${PYTHONPATH}:/rep/rabuhweidi/sft/SFTTraining-SemEval/src" python /rep/rabuhweidi/sft/SFTTraining-SemEval/src/blm/utils/eval.py \
    --model_path meta-llama/Llama-3.2-3B-Instruct \
    --test_data_path /rep/rabuhweidi/sft/SFTTraining-SemEval/input_folder/eval \
    --hf_token hf_***** \
    --tokenizer_name meta-llama/Llama-3.2-3B-Instruct \
    --output_predictions_path /rep/rabuhweidi/sft/SFTTraining-SemEval/output_folder/CICIoT_2023/logs_Results/predict_log_Llama3B_CICoT_baseline.csv \
    --test_sample_limit 2240 \
    --temperature  0 \
    --top_p 1 \
    --top_k 1

evaluation-after SFT: 
PYTHONPATH="${PYTHONPATH}:/rep/rabuhweidi/sft/SFTTraining-SemEval/src" python /rep/rabuhweidi/sft/SFTTraining-SemEval/src/blm/utils/eval.py \
    --model_path /rep/rabuhweidi/sft/SFTTraining-SemEval/output_folder/CICIoT_2023/model_Llama3B_CICIoT_2023 \
    --test_data_path /rep/rabuhweidi/sft/SFTTraining-SemEval/input_folder/eval \
    --hf_token hf_***** \
    --tokenizer_name /rep/rabuhweidi/sft/SFTTraining-SemEval/output_folder/CICIoT_2023/model_Llama3B_CICIoT_2023 \
    --output_predictions_path /rep/rabuhweidi/sft/SFTTraining-SemEval/output_folder/CICIoT_2023/logs_Results/predict_log_Llama3B_CICoT_SFT.csv \
    --test_sample_limit 2240

save on huggingface:// not work in server
CUDA_LAUNCH_BLOCKING=1 PYTHONPATH="${PYTHONPATH}:/rep/rabuhweidi/sft/SFTTraining-SemEval/src" python /rep/rabuhweidi/sft/SFTTraining-SemEval/src/blm/utils/save_to_hfhub.py \
    --model_path "/rep/rabuhweidi/sft/CICIoT_2023/Llama_3B_CICIoT_2023/model_Llama3B_CICIoT_2023" \
    --repo_name "RuwaYafa/Llama-3.2-3B-Instruct-CICIoT_2023" \
    --hf_token "hf_*****" \
    --your_name "Tarqeem" \
#    --private  # Remove this for public repo