import re
import pandas as pd
import numpy as np
random_seed = 42
np.random.seed(random_seed)

def extract_inner_keys(d, inner_keys=None):
    if inner_keys is None:
        inner_keys = set()
    for v in d.values():
        if isinstance(v, dict):
            inner_keys.update(key for key in v.keys() if key != "")
    return inner_keys

def replace_bpic2017(text):
    text = re.sub(r'W_', 'workflow: ', text)
    text = re.sub(r'A_', 'application: ', text)
    text = re.sub(r'O_', 'offer: ', text)
    text = re.sub(r'\(|\)', '', text)  
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def state_initialise(log, bos):
    data = {key: [key in bos for _ in range(len(log))] for key in bos}
    for key, value in data.items():
        log[key] = "unprocessed"
    return log


def assign_state(case, mappings):
    activity = case['Activity'].strip()

    changes = mappings[activity]
    if changes.keys() != "":
        for col, new_value in changes.items():
            if isinstance(new_value, dict):
                for subcol, subnew_value in new_value.items():
                    if subcol != "":
                        case[subcol] = subnew_value
            else:
                if col != "":
                    case[col] = new_value

    return case

def apply_mapping(group, bos, mappings):
    bos = list(bos)
    group = group.reset_index(drop=True)

    def broadcast_columns(group):
        first_row = group.iloc[i][bos]
        group[bos] = first_row
        return group

    if len(group) > 1:
        for i in range(len(group)):
            group.iloc[i] = assign_state(group.iloc[i], mappings)
            values_bos = group[bos].iloc[i].values
            group.loc[group.index[i]:, bos] = values_bos

    return group


def mark_next_activity(group):
    # mark next activity for each group
    group = group.sort_values(by='event_nr').reset_index(drop=True)
    if 'Activity_ori' in group.columns:

        group['label'] = group['Activity_ori'].shift(-1)
    else:
        group['label'] = group['Activity'].shift(-1)
    return group[:-1]

def filter_groups(group):
    return len(group) > 1
def gen_prefix_data(data, min_length, max_length, gap=1):

    def generate_features(group, max_features):
        n = len(group)

        if 'Activity_ori' in group.columns:
            act_sequence = group['Activity_ori'].tolist()
        else:
            act_sequence = group['Activity'].tolist()
        subseq_list = []
        end_index_list = []
        label_list = []

        for _ in range(10): 
            # start_index = np.random.randint(0, n-1)
            end_index = np.random.randint(1, n)
            start_index = end_index - max_features
            # end_index = start_index + max_features
            if start_index < 0:
                subseq = act_sequence[0:end_index]
                if len(subseq) < max_features:
                    subseq.extend(['end'] * (max_features - end_index))
            else:
                subseq = act_sequence[start_index:end_index]

            end_index_list.append(end_index-1)
            subseq_list.append(subseq)
            label_list.append(act_sequence[end_index] if end_index < n-1 else 'end')

        selected_rows = group.iloc[end_index_list]
        new_df = selected_rows.copy()
        new_df['label'] = label_list

        for i in range(max_features):
            col_name = f'act_{i + 1}'  
            new_df[col_name] = [subseq[i] for subseq in subseq_list]

        return new_df

    data = data.groupby('Case ID').filter(filter_groups)
    new_df = data.groupby('Case ID').apply(lambda x: generate_features(x, max_length)).reset_index(level=0, drop=True).reset_index()


    # dt_prefixes = data[data['case_length'] >= min_length].groupby('Case ID').head(min_length)
    # print(dt_prefixes)
    # dt_prefixes["prefix_nr"] = 1
    # dt_prefixes["orig_case_id"] = dt_prefixes['Case ID']
    # for nr_events in range(min_length + gap, max_length + 1, gap):
    #     tmp = data[data['case_length'] >= nr_events].groupby('Case ID').head(nr_events)
    #     tmp["orig_case_id"] = tmp['Case ID']
    #     tmp['Case ID'] = tmp['Case ID'].apply(lambda x: "%s_%s" % (x, nr_events))
    #     tmp["prefix_nr"] = nr_events
    #     dt_prefixes = pd.concat([dt_prefixes, tmp], axis=0)
    #
    # dt_prefixes['case_length'] = dt_prefixes['case_length'].apply(lambda x: min(max_length, x))

    return new_df

def gen_oop_prefix_data(data, min_length, max_length, gap=1):

    def generate_features(group, max_features):
        n = len(group)
        act_sequence = group['Activity'].tolist()
        subseq_list = []
        end_index_list = []

        for _ in range(10): 
            # start_index = np.random.randint(0, n-1)
            end_index = np.random.randint(1, n)
            start_index = end_index - max_features
            # end_index = start_index + max_features
            if start_index < 0:
                subseq = act_sequence[0:end_index]
                if len(subseq) < max_features:
                    subseq.extend(['end'] * (max_features - end_index))
            else:
                subseq = act_sequence[start_index:end_index]

            end_index_list.append(end_index-1)
            subseq_list.append(subseq)


        selected_rows = group.iloc[end_index_list]
        new_df = selected_rows.copy()


        for i in range(max_features):
            col_name = f'act_{i + 1}'  
            new_df[col_name] = [subseq[i] for subseq in subseq_list]

        return new_df

    data = data.groupby('Case ID').filter(filter_groups)
    new_df = data.groupby('Case ID').apply(lambda x: generate_features(x, max_length)).reset_index(level=0, drop=True).reset_index()

    return new_df

def generate_prefix_data(data, min_length, max_length, gap=1):
    # generate prefix data (each possible prefix becomes a trace)

    data['case_length'] = data.groupby('Case ID')['Activity'].transform(len)

    dt_prefixes = data[data['case_length'] >= min_length].groupby('Case ID').head(min_length)
    dt_prefixes["prefix_nr"] = 1
    dt_prefixes["orig_case_id"] = dt_prefixes['Case ID']
    for nr_events in range(min_length + gap, max_length + 1, gap):
        tmp = data[data['case_length'] >= nr_events].groupby('Case ID').head(nr_events)
        tmp["orig_case_id"] = tmp['Case ID']
        tmp['Case ID'] = tmp['Case ID'].apply(lambda x: "%s_%s" % (x, nr_events))
        tmp["prefix_nr"] = nr_events
        dt_prefixes = pd.concat([dt_prefixes, tmp], axis=0)

    dt_prefixes['case_length'] = dt_prefixes['case_length'].apply(lambda x: min(max_length, x))

    return dt_prefixes

def gen_agg(data, min_length, max_length, categorical_features):
    result_df = pd.DataFrame()

    for case_id, group in data.groupby('Case ID'):
       
        group = group.sort_values('event_nr')
        group = group[group['event_nr'] < max_length]

        for feature in categorical_features:
            for value in group[feature].unique():
                new_col_name = f'{feature}_{value}_count'
                group[new_col_name] = 0
                for idx in range(len(group)):
                    count = (group.loc[:group.index[idx], feature] == value).sum()
                    group.at[group.index[idx], new_col_name] = count

        #nap label
        group['nap_label'] = group['Activity'].shift(-1)
        group['nap_label'] = group['nap_label'].fillna("end")

        result_df = pd.concat([result_df, group], ignore_index=True)
    result_df.fillna(0, inplace=True)

    return result_df

def gen_order(data, min_length, max_length, categorical_features):
    result_df = pd.DataFrame()
    for case_id, group in data.groupby('Case ID'):
        group = group.sort_values('event_nr')
        group = group[group['event_nr'] < max_length]

        for feature in categorical_features:
            for value in group[feature].unique():
                new_col_name = f'{feature}_{value}_order'
                group[new_col_name] = 0
                for idx in range(len(group)):
                    value_series = group.loc[:group.index[idx], feature]
                    value_series = value_series.reset_index(drop=True)
                    value_series.index += 1  # Adjust index to start from 1
                    result_dict = {value: value_series[value_series == value].index.max() for value in value_series.unique()}
                    if value in result_dict.keys():
                        group.at[group.index[idx], new_col_name] = result_dict[value]
            # for value in group[feature].unique():
            #     group[new_col_name] = 0
            #     for idx in range(len(group)):
            #         count = (group.loc[:group.index[idx], feature] == value).sum()
            #         group.at[group.index[idx], new_col_name] = count


        #nap label
        group['nap_label'] = group['Activity'].shift(-1)
        group['nap_label'] = group['nap_label'].fillna("end")

    
        result_df = pd.concat([result_df, group], ignore_index=True)
    result_df.fillna(0, inplace=True)

    return result_df
