import os
import ast
import argparse
import json
import pandas as pd

from transformers import pipeline
import torch

from constant import *
import utilscon

#stabilize
import random
random.seed(42)
import numpy as np
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)  # if using CUDA
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt



example1 = """
{"application: Create Application Created"}
"""
answer1 = """
{"application: Create Application Created": {"application": "created"}}
"""
example2 = """
{"workflow: Complete application Deleted"}
"""
answer2 = """
{"workflow: Complete application Deleted": {"application": "complete", "application": "deleted"}}
"""
example3 = """
{"offer: Sent mail and online statechange"}
"""
answer3 = """
{"offer: Sent mail and online statechange": {"offer": "sent both"}}}
"""

example4 = """
{"application: Create Application Created"}
"""
answer4 = """
{"application: Create Application Created": {"application": "created"}}
"""
example5 = """
{"workflow: Complete application Deleted"}
"""
answer5 = """
{"workflow: Complete application Deleted": {"application": "complete", "application": "deleted"}}
"""
example6 = """
{"offer: Sent mail and online statechange"}
"""
answer6 = """
{"offer: Sent mail and online statechange": {"offer": "sent both"}}}
"""

prompt_llama = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert in process mining. Your task is to extract key business objects and their states from complex activity labels by mapping each activity label to a dictionary in the form of {{activity label: {{business object: status}}}}. 
A single activity label can contain multiple essential business objects and states. Business objects must be general and process-related.
Respond strictly in JSON format, providing fields for each activity. Ensure you understand the meaning and don't just extract directly. Output only the dictionary as the answer. Don't forget any of given activity labels.

Example activity labels: {example1}
Example answer: {answer1}

Example activity labels: {example2}
Example answer: {answer2}

Example activity labels: {example3}
Example answer: {answer3}

Example activity labels: {example4}
Example answer: {answer4}

Example activity labels: {example5}
Example answer: {answer5}

Example activity labels: {example6}
Example answer: {answer6}


<|eot_id|><|start_header_id|>user<|end_header_id|>
"""

suffix_prompt_llama = """
<|eot_id|><|start_header_id|>expert<|end_header_id|>
"""

prompt_mistral = f"""
You are an expert in process mining. Your task is to extract key business objects and their states from complex activity labels after <<<>>> by mapping each activity label to a dictionary in the form of {{activity label: {{business object: status}}}}.
A single activity label can contain multiple essential business objects and states. Business objects must be general and process-related.
Respond strictly in JSON format, providing fields for each activity. Ensure you understand the meaning and don't just extract directly. Output only the dictionary as the answer. Don't forget any of given activity labels.

####
Here are some examples:

Activity labels: {example1}
Answer: {answer1}
Activity labels: {example2}
Answer: {answer2}
Activity labels: {example3}
Answer: {answer3}

###

<<<
Activity labels:  

"""

suffix_prompt_mistral = """
>>>"""

def gen_act_labels(aug_log):
    # generate uniform activity labels set from each event log
    labels = set(aug_log['Activity'].unique())
    print(f"length of labels: {len(labels)}")
    return labels

def gen_prompt(labels, model):
    # generate prompt for each log and each model
    prompt = ""
    if "llama" in model:
        prompt = prompt_llama + str(labels) + suffix_prompt_llama
    elif "mistral" in model:
        prompt = prompt_mistral + str(labels) + suffix_prompt_mistral

    print(prompt)
    return prompt


def gen_map(prompt, device, llm_model_name):
    text_generator = pipeline(
        "text-generation",
        model=llm_model_name,
        model_kwargs={
            "torch_dtype": torch.float16,
            "quantization_config": {"load_in_8bit": True},
            "low_cpu_mem_usage": True,
        },
    )
    input_tokens = text_generator.tokenizer(prompt, return_tensors="pt")
    print(f"Number of tokens in prompt: {len(input_tokens['input_ids'][0])}")

    response = text_generator(
        prompt,
        max_new_tokens=1000,
        num_return_sequences=1,
        temperature=0.6,
        top_k=50,
        top_p=0.95,
        return_full_text=False,
        pad_token_id=text_generator.tokenizer.eos_token_id
    )
    generated_text = response[0]['generated_text']
    print(generated_text)
    count = 0
    for char in generated_text:
        if char == '{':
            count += 1

    print("Number of keys:", count-1)

    return response[0]['generated_text']

def gen_df(log, map):
    bos = utilscon.extract_inner_keys(map)
    log = utilscon.state_initialise(log, bos)
    object_stated_log = log.groupby('Case ID', group_keys=False).apply(utils.apply_mapping, bos, map).reset_index(
        drop=True)
    return object_stated_log


def semantic_extraction_llm(lm_model, data_name):
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    data_path = DATA_PATHS[data_name]
    log = pd.read_csv(data_path, sep=";")
    activity_labels = gen_act_labels(log)
    prompt = gen_prompt(activity_labels, lm_model)
    map_str = gen_map(prompt, device, lm_model)
    map_str = map_str.replace("'", "\"")
    try:
        map = ast.literal_eval(map_str)
        with open(f'{data_name}_llama.json', 'w') as json_file:
            json.dump(map, json_file, indent=4, ensure_ascii=False)
    except ValueError:
        print("Error: Unable to convert string to dictionary.")

    df = gen_df(log, map_str)
    print(df)
    return map_str


def semantic_annotation(data_name, pattern):
    data_path = DATA_PATHS[data_name]
    log = pd.read_csv(data_path, sep=";")
    # business objects extraction
    set_activities = log['Activity'].unique().tolist()
    print(set_activities)
    print(len(set_activities))
    map_path = MAP_PATHS[data_name]
    with open(f'{map_path}_{pattern}.json', 'r') as f:
        mappings = json.load(f)
    bos = utilscon.extract_inner_keys(mappings)
    log = utilscon.state_initialise(log, bos)
    object_stated_log = log.groupby('Case ID', group_keys=False).apply(utilscon.apply_mapping, bos, mappings).reset_index(
        drop=True)
    print(object_stated_log)
    object_stated_log.to_csv(f'datas/{data_name}_{pattern}.csv')


def predict(task, ori, state, pattern):
    data_name = "production"
    min_prefix_length = 3
    max_prefix_length = 20

    if ori is True:
        data_path = DATA_PATHS[data_name]
        log = pd.read_csv(data_path, sep=";")
    else:
        if pattern == "hard":
            data_path = DATA_PATHS_H[data_name]
        elif method == "llm":
            data_path = DATA_PATHS_L[data_name]
        else:
            raise ValueError("choose either 'hard' or 'llm'")
        log = pd.read_csv(data_path, sep=",")

    print(data_path)

    categorical_features = ['Activity']
    if "bpic2017" in data_name:
        time_feat = 'time:timestamp'
    else:
        time_feat = 'Complete Timestamp'

    if state is False:
        log = utilscon.gen_agg(log, min_prefix_length, max_prefix_length, categorical_features)
    else:
        log = log[log['event_nr'] < max_prefix_length]

    if task == "nap":
        log['nap_label'] = log.groupby('Case ID')['Activity'].shift(-1)
        log['nap_label'] = log['nap_label'].fillna('end')
        log['label'] = log['nap_label']
        log.drop(columns=['nap_label'], inplace=True)
        y = log['label']
        num_classes = np.unique(y).size
        cls = XGBClassifier(objective='multi:softmax', num_class=num_classes, seed=42)

    all_columns = log.columns
    if ori is True:
        del_columns = [col for col in all_columns if not col.endswith('count')]
        del_columns = [col for col in del_columns if col not in ['Case ID', 'label', 'event_nr', time_feat]]
        log = log.drop(columns=del_columns)
    else:
        ori_log_path = DATA_PATHS[data_name]
        ori_log = pd.read_csv(ori_log_path, sep=";", nrows=0)
        ori_features = ori_log.columns
        semantic_features = [col for col in all_columns if col not in ori_features]
        if "Unnamed: 0" in semantic_features:
            semantic_features.remove(
                "Unnamed: 0")
        del_columns = [col for col in all_columns if not col.endswith('order')]
        del_columns = [col for col in del_columns if col not in ['Case ID', 'label', 'event_nr', time_feat]]
        del_columns = [col for col in del_columns if col not in semantic_features]
        log = log.drop(columns=del_columns)
    label_encoders = {}
    cat_features_to_encode = []
    for col in log.columns:
        if log[col].dtype == 'object':
            cat_features_to_encode.append(col)
    for col in cat_features_to_encode:
        label_encoders[col] = LabelEncoder()
        log[col] = label_encoders[col].fit_transform(log[col].astype(str))

    if task == "oop":
        if 'nap_label' in log.columns:
            log.drop(columns=['nap_label'], inplace=True)
        cls = XGBClassifier(objective='binary:logistic', seed=42) #tree_method = "hist", device = "cuda"


    #groupby case
    unique_cases = log['Case ID'].unique()
    train_cases, test_cases = train_test_split(unique_cases, test_size=0.2, random_state=42)
    X_train = log[log['Case ID'].isin(train_cases)]
    X_test = log[log['Case ID'].isin(test_cases)]
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(X_train['label'])
    y_test = label_encoder.fit_transform(X_test['label'])
    X_train = X_train.drop(columns=['label'])
    print(f'train length: {len(y_train)}')
    print(f'test length: {len(y_test)}')
    X_train = X_train.drop(columns=['Case ID', time_feat])
    X_test = X_test.drop(columns=['Case ID', time_feat])
    cls.fit(X_train, y_train)
    results = []

    for event_nr, group in X_test.groupby('event_nr'):
        if event_nr < min_prefix_length or event_nr > max_prefix_length:
                continue
        X_group = group.drop(columns=['label'])
        y_group = group['label']
        sample_count = len(group)
        if task == "oop":
            y_pred_proba = cls.predict_proba(X_group)[:, 1]
            try:
                auc_roc = roc_auc_score(y_group, y_pred_proba)
            except ValueError:
                pass
            results.append({'event_nr': event_nr, 'auc_roc_score': auc_roc, 'sample_count': sample_count})
        elif task == "nap":
            y_pred_group = cls.predict(X_group)
            acc = accuracy_score(y_group, y_pred_group)
            f1 = f1_score(y_group, y_pred_group, average='macro')
            results.append({'event_nr': event_nr, 'accuracy': acc, 'f1_macro': f1, 'sample_count': sample_count})
    results_df = pd.DataFrame(results)
    print("Results by group:")
    print(results_df)


    if task == "oop":
        weighted_avg_auc_roc = (results_df['auc_roc_score'] * results_df['sample_count']).sum() / results_df[
            'sample_count'].sum()
        print("\nWeighted Average AUC-ROC Score:", weighted_avg_auc_roc)

    elif task == "nap":
        weighted_acc = (results_df['accuracy'] * results_df['sample_count']).sum() / results_df[
            'sample_count'].sum()

        print("\nWeighted Average Accuracy:", weighted_acc)

    #overal accuracy
    X_test = X_test.drop(columns=['label'])
    y_pred = cls.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"overal Accuracy: {accuracy}")
    f1_macro = f1_score(y_test, y_pred, average='macro')
    print("Macro-Averaged F1 Score:", f1_macro)
    print(len(y_pred))

    plot_importance(cls)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A simple command line interface')
    parser.add_argument('pattern', choices=PATTERNS, help='The way to extract business objects and statuses, either "hard" or "llm"')
    parser.add_argument('lm_model', help='The huggingface language model to use (e.g. "meta-llama/Meta-Llama-3-8B")')
    parser.add_argument('data_name', choices=DATAS, help='The dataset to use (e.g. "production")')
    args = parser.parse_args()
    semantic_extraction_llm(lm_model="meta-llama/Meta-Llama-3-8B", data_name="production")  #You need a GPU with at least 16GB of VRAM and 16GB of system RAM to run Llama 3-8B
    semantic_annotation(data_name="production", pattern="hard")
    predict(task="oop", ori=False, state=False, pattern="hard")



