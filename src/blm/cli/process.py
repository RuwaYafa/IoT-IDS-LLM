

import os
import sys #rfa
# sys.path.append(r'D:\Code\LLMTraining-main\src') #rfa

import logging
import argparse
from datasets import  load_dataset,Dataset, DatasetDict, Features, Value, Sequence, ClassLabel
from blm.utils.helpers import logging_config
import pyarrow as paPyright
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

def get_data_records(dataset):
    
    data_records = []
    label = {
        'DDoS': 0,
        'DoS': 1, 
        'Benign':2,
        'Unknown': 3
    }
    i=0
    # print(f'dataset----------------{dataset}\n\n\n\n\n')
    for row in dataset:#df.iterrows():
        
        sentence = ", ".join([f"{col}={row[col]}" for col in dataset.features if col != "target"])
        relation = label[row["target"]]
        if i<10: 
            # print(relation)
            i = i+1
        data_records.append({"sentence": sentence, "relation": relation})
        
    return data_records

def save_dataset(dataset, output_path, n):
    """
    Convert the dataset into messages as follows, each training example will
    follow the following format:
      {
        "messages": [
                    {"content": "<SYSTEMPROMPT>", "role": "system"}, # it depends on the model prompt structure
                    {"content": "<INSTRUCTIONS>", "role": "user"},
                    {"content": "<RESPONSE>", "role": "assistant"}
                    ]
      }
    :param dataset: Dataset
    :param output_path: str - output path to save dataset to
    :param n: int - number of training examples to sample
    :return: Dataset
    """

    # Open the system prompt file and read its content #mon
    with open('/rep/rabuhweidi/sft/SFTTraining-SemEval/src/blm/prompts/system_prompt.txt', 'r') as file:
        system_prompt = file.read()

    # Open the user prompt file and read its content #mon
    with open('/rep/rabuhweidi/sft/SFTTraining-SemEval/src/blm/prompts/user_prompt.txt', 'r') as file:
        user_prompt = file.read()

    # logger.info(f'Total train examples: {dataset["train"][0:3]}')
    # logger.info(f'Total test examples: {dataset["test"][0:3]}')

    data_records_train = get_data_records(dataset['train'])
    train_messages = [{"messages": [{"content": system_prompt, "role": "system"}, 
                                    # If we have multi class just add them as , instruction=e['instruction'] etc..
                                    {"content": user_prompt.format(sentence=e['sentence']), "role": "user"}, 
                                    {"content": str(e["relation"]), "role": "assistant"}]} 
                      for e in data_records_train
                      ]
    data_records_test = get_data_records(dataset['test'])
    eval_messages= [{"messages": [{"content": system_prompt, "role": "system"}, 
                                  {"content": user_prompt.format(sentence=e['sentence']), "role": "user"}, 
                                  {"content": str(e["relation"]), "role": "assistant"}]} 
                    for e in data_records_test
                    ]
    
    #End RFA
    # print("Train message",train_messages[0])
    # print("eval message",eval_messages[0])

    ds = DatasetDict({
        # "train": Dataset.from_list(train_messages[:int(n*0.8)]),
        # "eval": Dataset.from_list(eval_messages[int(n*0.8):n])

        "train": Dataset.from_list(train_messages), #rfa
        "eval": Dataset.from_list(eval_messages) #rfa
    })
    
    # print(f'train msg----------------{train_messages}')

    logger.info(f'Total train examples: {ds["train"][0:3]}')
    logger.info(f'Total eval examples: {ds["eval"][0:3]}')


    print("dsdsds", ds["train"][0])

    ds.save_to_disk(output_path)
    return ds


def main(args):
    # dataset = load_dataset("arbml/CIDAR") # change the Dataset / from huggingface or from data folder! #mon ds: #Kamyar-zeinalipour/ArabicSense
    # dataset = dataset['train'].train_test_split(test_size=0.2)
   
    # print("Dataset____",dataset)
    # print("end dataset")
    # print("Dataset____",dataset["train"])
    # print("Dataset____",dataset["test"])

    dataset = load_dataset(
    "csv", 
    data_files={
        "train": "/rep/rabuhweidi/sft/SFTTraining-SemEval/data/train.csv", 
        "test": "/rep/rabuhweidi/sft/SFTTraining-SemEval/data/test.csv", 
        # "validation": "/content/drive/MyDrive/LLMTraining-main/data/validation.csv"
    }
    )
    

    ds = save_dataset(dataset, args.output_path, args.n)
    logger.info(f"Total training examples length: {len(ds['train'])}")
    logger.info(f"Total eval examples length: {len(ds['eval'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--log_file", type=str, help="Log file path")
    parser.add_argument("--output_path", type=str, help="Data output path")
    parser.add_argument("--n", type=int, default=5, help="Number of training examples to sample")
    args = parser.parse_args()


    os.makedirs(os.path.join(args.output_path), exist_ok=True)
    os.makedirs(os.path.join("log"), exist_ok=True)
    logging_config("/rep/rabuhweidi/sft/SFTTraining-SemEval/log/log.log")
    # logging_config("/content/drive/MyDrive/SFTTraining-SemEval-main/processing.log")

    main(args)
