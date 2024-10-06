from typing import Iterator
from datasets import load_dataset, concatenate_datasets
import datasets
from torch.utils.data import Dataset, IterableDataset, DataLoader
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import os
from transformers import AutoTokenizer
import transformers
from functools import partial
from prompt_maker import PromptMaker
import glob
import copy
import json
import random
from tqdm import tqdm
import jsonlines
import multiprocessing
from time import time
from conversation_template import PRESET_TEMPLATES

def flat_map_function(element):
    # Replace with your actual logic to return a list
    return [element, element * 2]

# Function to flatten the list
def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]



class JsonDataset(Dataset):
    def __init__(self,
            json_data: Union[os.PathLike, List[Dict]],
            transform: Callable = None, 
            shuffle: bool = True,
            train=False,
            chunk_long_text=False,
            lmflow_format=False
        ):
        
        json_filenames = glob.glob(json_data)

        self.source_dict = {fn: idx for fn, idx in zip(json_filenames, range(len(json_filenames)))}
        self.data_files = json_filenames
        data_list = []

        print(self.source_dict)

        for fn, idx in self.source_dict.items():
            if fn.endswith('.json'):
                data = json.load(open(fn))
            elif fn.endswith('.jsonl'):
                data = [i for i in jsonlines.Reader(open(fn))]
            else:
                raise ValueError('Input File Is Either Json or Jsonline')
            if lmflow_format:
                contents=data['instances']
                for d in contents: d["source"] = idx
                data_list.extend(contents)
            else:
                for d in data: d["source"] = idx
                data_list.extend(data)
        
        self.data=data_list
        # self.data=datasets.Dataset.from_list(data_list)
        # print(self.data)
        if shuffle:
            # self.data=self.data.shuffle()
            random.shuffle(self.data)
        self.transform=transform
        if transform:
            # self.data=self.data.map(self.transform, batched=True, batch_size=1, num_proc=5)
            chunked=[]
            if chunk_long_text:
                for i in tqdm(range(len(self.data)), miniters=500):
                    trans=transform(self.data[i])
                    chunked.extend(trans)
                self.data=chunked
            else:
                for i in tqdm(range(len(self.data)), miniters=500):
                    self.data[i] = {**self.data[i], **transform(self.data[i])}
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.transform:
            return {'input_ids': self.data[index]['input_ids'], 
                    'source': self.data[index]['source'], 
                    'labels': self.data[index]['labels']}
        else:
            return self.data[index]


class JsonDataset_deprecate(Dataset):
    def __init__(self,
            json_data: Union[os.PathLike, List[Dict]],
            transform: Callable = None, 
            shuffle: bool = True,
            seed=42,
            train=False,
        ):
        
        json_filenames = glob.glob(json_data)

        self.source_dict = {fn: idx for fn, idx in zip(json_filenames, range(len(json_filenames)))}
        self.data_files = json_filenames
        data_list = []

        print(self.source_dict)

        for fn, idx in self.source_dict.items():
            data = load_dataset(
                "json", 
                data_files=fn, 
                split="train", 
                streaming=False, 
                keep_in_memory=True
            )
            data = data.map(lambda examples: {"source": idx})
            data_list.append(data)
        
        self.data=concatenate_datasets(data_list, split="train")

        if shuffle:
            self.data = self.data.shuffle(seed=seed)
        self.transform=transform
        # if transform:
            # self.data = self.data.map(transform)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.transform:
            source=self.data[index]['source']
            res=self.transform(self.data[index])
            return {'input_ids': res['input_ids'], 
                    'source': source, 
                    'labels': res['labels']}
        else:
            return self.data[index]

class StreamJsonDataset(IterableDataset):
    """
    DO NOT USE
    stream for large scale data
    first shuffle data sources, then shuffle data in the shuffle buffer
    """

    def __init__(self,
            json_data: Union[os.PathLike, List[Dict]],
            transform: Callable = None, 
            shuffle: bool = True,
            shuffle_buffer_size: int = 1000,
            seed=42,
            train=False,
        ):

        if train:
            json_filenames = glob.glob(json_data)
        else:
            json_filenames = [json_data]
        
        self.source_dict = {fn: idx for fn, idx in zip(json_filenames, range(len(json_filenames)))}
        
        self.data_files = json_filenames

        dataiter_list=[]
        for fn, idx in self.source_dict.items():
            dataiter = load_dataset(
                "json", 
                data_files=fn, 
                split="train", 
                streaming=True, 
                keep_in_memory=True
            )
            dataiter = dataiter.map(lambda x, source: {"source": source}, fn_kwargs={"source": idx})
            dataiter_list.append(dataiter)
            
        self.dataiter=concatenate_datasets(dataiter_list)
        print(type(self.dataiter))
        if shuffle:
            self.dataiter = self.dataiter.shuffle(buffer_size=shuffle_buffer_size, seed=seed)

        if transform:
            self.dataiter = self.dataiter.map(transform)

    def __iter__(self):
        return iter(self.dataiter)

def tokenize_prompt(data_point: Dict = None,
    max_length: int = 256,
    tokenizer = None,
    prompt_maker = None,
    response_loss_only: bool = True,
    padding: Union[bool, str] = False,
    truncation: bool = True,
):
    assert prompt_maker is not None, "please provide prompt_maker"
    assert tokenizer is not None, "please provide tokenizer"

    full_text = prompt_maker.get_full(data_point)
    full_tokenized=tokenizer(full_text, max_length=max_length, 
                        truncation=truncation, padding=padding, 
                        add_special_tokens=True, )["input_ids"]
    ## WARNING some tokenizer may not automatically add eos token

    if full_tokenized[-1] != tokenizer.eos_token_id: 
        full_tokenized = full_tokenized + [tokenizer.eos_token_id]
    
    if not response_loss_only:
        return {"input_ids": full_tokenized, "labels": copy.deepcopy(full_tokenized)}
    else:
        input_token=tokenizer(prompt_maker.get_input(data_point), 
                              max_length=max_length, 
                              truncation=truncation, padding=padding, 
                              add_special_tokens=False, )["input_ids"]
        labels = [-100] * len(input_token) + full_tokenized[len(input_token):]
        # attention_mask = [1] * len(full_tokenized)
        return {"input_ids": full_tokenized, "labels": labels}

def tokenize_conversion(data_point: Dict = None,
    max_length: int = 256,
    tokenizer = None,
    prompt_maker = None,
    response_loss_only: bool = True,
    padding: Union[bool, str] = False,
    truncation: bool = True,
):
    assert prompt_maker is None, "no need to use prompt maker"
    assert tokenizer is not None, "please provide tokenizer"
    # print(data_point)
    # sharegpt data format
    conversations = data_point['items']
    # assert conversations[0]['from'] == 'human' and len(conversations) %2 ==0, "check share gpt data format"

    input_ids = []
    labels = []
    for c in conversations:

        if c['from'] == 'human':
            text = "###Human: " + c['value']
            tokens = tokenizer(text, max_length=999999999999, 
                        truncation=truncation, padding=padding, 
                        add_special_tokens=False, )["input_ids"]
            # if len(tokens) == 0 or tokens[-1] != tokenizer.eos_token_id: 
                # tokens += [tokenizer.eos_token_id]
            input_ids += tokens
            labels += [-100] * len(tokens)
        elif c['from'] == 'gpt':
            text = "###Assistant: " + c['value']
            tokens = tokenizer(text, max_length=999999999999, 
                        truncation=truncation, padding=padding, 
                        add_special_tokens=True, )["input_ids"]

            if len(tokens) == 0 or tokens[-1] != tokenizer.eos_token_id: 
                tokens += [tokenizer.eos_token_id]
            input_ids += tokens
            labels += tokens
        else:
            raise NotImplementedError('Wrong from id in share gpt data')
        if len(input_ids) >= max_length:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]
            break


    return {"input_ids": input_ids, "labels": copy.deepcopy(labels)}

def tokenize_conversion_lmflow(data_point: Dict = None,
    max_length: int = 256,
    tokenizer = None,
    prompt_maker = None,
    response_loss_only: bool = True,
    padding: Union[bool, str] = False,
    truncation: bool = True,
    chat_template: str = 'llama3',
):
    assert prompt_maker is None, "no need to use prompt maker"
    assert tokenizer is not None, "please provide tokenizer"

    conversation_template = PRESET_TEMPLATES[chat_template]
    encoded_conversation = conversation_template.encode_conversation(
                tokenizer=tokenizer,
                messages=data_point['messages'],
                system=None,
                tools=None,
            )
    input_ids, labels = [], []
    for turn_idx, (user_input, assistant_result) in enumerate(encoded_conversation):
        input_ids += user_input + assistant_result
        
        if not response_loss_only:
            labels += user_input + assistant_result
        else:
            labels += [-100] * len(user_input) + assistant_result
    return {"input_ids": input_ids[:max_length], "labels": labels[:max_length]}

def tokenize_conversion_lmflow_(data_point: Dict = None,
    max_length: int = 256,
    tokenizer = None,
    prompt_maker = None,
    response_loss_only: bool = True,
    padding: Union[bool, str] = False,
    truncation: bool = True,
):
    assert prompt_maker is None, "no need to use prompt maker"
    assert tokenizer is not None, "please provide tokenizer"
    # print(data_point)
    # sharegpt data format
    conversations = data_point['messages']
    # assert conversations[0]['from'] == 'human' and len(conversations) %2 ==0, "check share gpt data format"

    input_ids = []
    labels = []
    for c in conversations:

        if c['role'] == 'user':
            text = c['content']
            tokens = tokenizer(text, max_length=999999999999, 
                        truncation=truncation, padding=padding, 
                        add_special_tokens=False, )["input_ids"]
            # if len(tokens) == 0 or tokens[-1] != tokenizer.eos_token_id: 
                # tokens += [tokenizer.eos_token_id]
            input_ids += tokens
            labels += [-100] * len(tokens)
        elif c['role'] == 'assistant':
            text = c['content']
            tokens = tokenizer(text, max_length=999999999999, 
                        truncation=truncation, padding=padding, 
                        add_special_tokens=True, )["input_ids"]

            if len(tokens) == 0 or tokens[-1] != tokenizer.eos_token_id: 
                tokens += [tokenizer.eos_token_id]
            input_ids += tokens
            labels += tokens
        else:
            raise NotImplementedError('Wrong from id in share gpt data')
        if len(input_ids) >= max_length:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]
            break


    return {"input_ids": input_ids, "labels": copy.deepcopy(labels)}

def tokenize_text_only(data_point: Dict = None,
    max_length: int = 256,
    tokenizer = None,
    prompt_maker = None,
    response_loss_only: bool = True,
    padding: Union[bool, str] = False,
    truncation: bool = True,
):
    assert prompt_maker is None, "no need to use prompt maker"
    assert tokenizer is not None, "please provide tokenizer"

    text=data_point['text']
    input_ids = []
    labels = []

    tokens = tokenizer(text, max_length=999999999999, 
                        truncation=truncation, padding=padding, 
                        add_special_tokens=True, )["input_ids"]
    
    if len(tokens) == 0 or tokens[-1] != tokenizer.eos_token_id: 
        tokens += [tokenizer.eos_token_id]

    res=[]
    for i in range(0, len(tokens), max_length):
        res.append({"input_ids": tokens[i : i + max_length], "labels": copy.deepcopy(tokens[i : i + max_length]), "source": data_point["source"]})

    return res

if __name__ == '__main__':

    # data_file = "/home/panxingyuan/bilevel_llm/data/train/alpaca_data.json"
    data_file = "/u/xpan2/projects/scalebio/LLM-Finetuning/data_lmflow/val.json"
    tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen2-7B")
    print(tokenizer)
    shuffled_iterable_dataset = JsonDataset(
        data_file, 
        shuffle=True, 
        train=True, 
        lmflow_format=True,
        transform=partial(tokenize_conversion_lmflow, tokenizer=tokenizer, max_length=128, chat_template='qwen2'),
    )

    print(shuffled_iterable_dataset[-1])
    print(tokenizer.decode(shuffled_iterable_dataset[-1]['input_ids']))
    # print(tokenizer.decode(shuffled_iterable_dataset[-1]['labels']))

   
    # for i in range(100):
    #     print(f'----------------------------------------sample {i}--------------------------------------------')
    #     print(len(shuffled_iterable_dataset[i]['input_ids']))
    # l=0.
    # for i in shuffled_iterable_dataset:
    #     l+=len(i['input_ids'])
    # print(l/len(shuffled_iterable_dataset))

    # data={
    #     "items": [
    #         {
    #             "from": "human",
    #             "value": "Write a Python program that simulates a simple calculator. It should take two operands and an operator (+, -, *, /) from the user and return the result."
    #         },
    #         {
    #             "from": "gpt",
    #             "value": "Certainly! Below is a simple Python program that acts as a calculator. It prompts the user for two numbers and an operator, then performs the calculation based on the operator provided.\n\n```python\ndef calculate(operand1, operand2, operator):\n    \"\"\"Perform calculation based on the operator.\"\"\"\n    if operator == '+':\n        return operand1 + operand2\n    elif operator == '-':\n        return operand1 - operand2\n    elif operator == '*':\n        return operand1 * operand2\n    elif operator == '/':\n        try:\n            return operand1 / operand2\n        except ZeroDivisionError:\n            return \"Error: Division by zero is not allowed.\"\n    else:\n        return \"Invalid operator\"\n\ndef main():\n    print(\"Simple Calculator\")\n    try:\n        # Taking input from the user\n        num1 = float(input(\"Enter first number: \"))\n        num2 = float(input(\"Enter second number: \"))\n        operator = input(\"Enter operator (+, -, *, /): \")\n\n        # Performing calculation\n        result = calculate(num1, num2, operator)\n\n        # Displaying the result\n        print(f\"Result: {result}\")\n    except ValueError:\n        print(\"Invalid input: Please enter a number.\")\n\nif __name__ == \"__main__\":\n    main()\n```\n\nThis program defines a `calculate` function that takes two operands and an operator, performs the calculation, and returns the result. The `main` function handles user input, calls the `calculate` function with the user's inputs, and prints the result. It also includes basic error handling for invalid numeric inputs and division by zero.\n\nTo use this calculator, simply run the program. It will prompt you to enter the first number, the second number, and the operator. After entering these, it will display the result of the calculation."
    #         },
    #         {
    #             "from": "human",
    #             "value": "Expand your calculator program to handle expressions with more than two operands (e.g., 2 + 3 * 4) using standard operator precedence without using the `eval` function or any libraries."
    #         },
    #         {
    #             "from": "gpt",
    #             "value": "Handling expressions with more than two operands and respecting operator precedence without using `eval` or external libraries requires implementing a basic expression parser. Below is a simplified version that supports addition, subtraction, multiplication, and division with standard operator precedence.\n\n```python\ndef parse_expression(expression):\n    \"\"\"Parse and evaluate the expression respecting operator precedence.\"\"\"\n    # Splitting by spaces for simplicity; this requires the user to input expressions with spaces (e.g., \"2 + 3 * 4\")\n    tokens = expression.split()\n    \n    # Convert numbers to floats and keep operators as is\n    tokens = [float(token) if token.isdigit() else token for token in tokens]\n    \n    # Handling multiplication and division first\n    i = 0\n    while i < len(tokens):\n        if tokens[i] == '*':\n            tokens[i-1] = tokens[i-1] * tokens[i+1]\n            del tokens[i:i+2]\n        elif tokens[i] == '/':\n            tokens[i-1] = tokens[i-1] / tokens[i+1]\n            del tokens[i:i+2]\n        else:\n            i += 1\n    \n    # Handling addition and subtraction\n    i = 0\n    while i < len(tokens):\n        if tokens[i] == '+':\n            tokens[i-1] = tokens[i-1] + tokens[i+1]\n            del tokens[i:i+2]\n        elif tokens[i] == '-':\n            tokens[i-1] = tokens[i-1] - tokens[i+1]\n            del tokens[i:i+2]\n        else:\n            i += 1\n    \n    return tokens[0]\n\ndef main():\n    print(\"Advanced Calculator\")\n    expression = input(\"Enter expression (e.g., 2 + 3 * 4): \")\n    try:\n        result = parse_expression(expression)\n        print(f\"Result: {result}\")\n    except Exception as e:\n        print(f\"Error: {e}\")\n\nif __name__ == \"__main__\":\n    main()\n```\n\nThis program introduces a `parse_expression` function that takes a string expression, splits it into tokens, and processes multiplication/division and addition/subtraction in separate passes to respect operator precedence. It's a basic implementation and assumes that the user inputs expressions with spaces between numbers and operators for simplicity. \n\nThis approach doesn't handle parentheses or more complex expressions with nested operations. Implementing full parsing for such cases would require a more sophisticated parser, possibly using the shunting-yard algorithm or constructing a parse tree."
    #         }
    #     ],
    #     "category": "coding"
    # }
    # res = tokenize_conversion(data, tokenizer=tokenizer, max_length=1024)
    # inp=tokenizer.decode(res['input_ids'])

    # print(len(res['input_ids']))
    # print(inp)