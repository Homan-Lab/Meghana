import pandas as pd
import pdb
from collections import Counter
import json
import os
import numpy as np

from helper_functions import sentence_embedding, convert_data_pldl_experiments, generate_data_bert, save_to_json, create_folder, generate_sentence_embedding_only,move_disco_embed_to_co

import sys

from tqdm import tqdm

from sklearn.model_selection import train_test_split
use_original_splits = False
sys.path.insert(0, 'utils/')
from tqdm import tqdm
from sklearn.model_selection import train_test_split
use_original_splits = False

def main():

    # raw_input_file = "/Users/meghanapg/Desktop/crowdopinion/prosocial-dialog.json"
    #####################
    folder = "./prosocial/DisCo"
    co_path = "./prosocial/co/"
    #########################
    raw_input_file = "casual_annotations.csv"
    # raw_input_file = "needs-caution_queries.csv"
    # raw_input_file = "needs-intervention_queries.csv"
    safety_label =['casual', 'possiblyneedscaution', 'probablyneedscaution', 'needscaution', 'needsintervention']

    # foldername
    # create_folder
    _id = "prosocial"
    # foldername1 = "datasets/prosocial/processed/modelling_annotator"
    # foldername2 = "datasets/prosocial/processed/modelling_annotator_nn"
    # foldername3 = "datasets/prosocial/processed/pldl"

    foldername1 = "data/prosocial/processed/modelling_annotator"
    foldername2 = "data/prosocial/processed/modelling_annotator_nn"
    foldername3 = "data/prosocial/processed/pldl"
    #######################################
        # worker_id_mt
    #######################################
    create_folder(foldername1)
    create_folder(foldername2)
    create_folder(foldername3)
    ###################################
    # dialogue_id and response_id in the dataset
    # dfs_combine = dfs_combine.rename(columns = {"rater_id":"annotator_id"})

    # dfs_combine = dfs_combine.rename(columns = {"item_id":"comment_id"})

    ###################################
    dfs_combine = pd.read_csv(raw_input_file)

    dfs_combine.drop_duplicates(inplace=True)

    # The line `# dfs_combine = dfs_combine.rename(columns = {"item_id":"comment_id"})` is a
    # commented-out line in the code. It appears to be a potential operation to rename a column in the
    # DataFrame `dfs_combine` from "item_id" to "comment_id". However, since it is commented out with
    # a `#` at the beginning, it is not currently being executed as part of the code.
    dfs_combine = dfs_combine.rename(columns = {"query":"message"})

    # dfs_combine = process_prompts(dfs_combine)


    dfs_combine = process_prompts(dfs_combine)


    label_dict = {index: safety_label[index] for index in range(0, len(safety_label))}

    dfs_combine['label'] = dfs_combine['safety_label'].astype('category')
    dfs_combine['label_vector'] = dfs_combine['label'].cat.codes
    cats = dfs_combine.label.astype('category')


    import pdb
    dfs_combine['Mindex'] = dfs_combine.index
    dfs_combine['comment_id'] = dfs_combine['Mindex']


    # dfs_combine["comment_id"] = range(len(dfs_combine))

    # Dropping the Category column as it is not of any importance for the project
    # dfs_combine = dfs_combine.drop(columns = ['rots'])
    # dfs_combine = dfs_combine.drop(columns = ['safety_annotations_reasons'])

    # data_items = pd.unique(dfs_combine['comment_id'])

    path = foldername1 + "/" + _id + "_annotations.json"


    dfs_combine.to_json(path, orient="split")
    # train, test and dev items
    # 50/25/25: train/test/dev
    train_items, dev_items = train_test_split(dfs_combine, test_size=0.4)
    dev_items, test_items = train_test_split(dev_items, test_size=0.5)

    dfs_dev = dfs_combine[dfs_combine.isin(dev_items)]
    path = foldername1 + "/" + _id + "_dev.json"
    dfs_dev.to_json(path, orient='split', index=False)

    dfs_train = dfs_combine[dfs_combine.isin(dev_items)]
    path = foldername1 + "/" + _id + "_train.json"
    dfs_train.to_json(path, orient='split', index=False)

    dfs_test = dfs_combine[dfs_combine.isin(test_items)]
    path = foldername1 + "/" + _id + "_test.json"
    dfs_test.to_json(path, orient='split', index=False)

    pdb.set_trace()

    # dfs_combine.reset_index(drop=True, inplace=True)
    # annotators_parsed.reset_index(drop=True, inplace=True)
    # dfs_combine = pd.concat([dfs_combine, annotators_parsed], axis=1)
    #
    # dfs_combine.reset_index(drop=True, inplace=True)
    # annotators_parsed.reset_index(drop=True, inplace=True)
    # dfs_combine = pd.concat([dfs_combine, annotators_parsed], axis=1)


    # path = foldername1 + "/" + _id + "_annotations.json"
    # ds_df = dfs_combine
    # ds_df.to_json(path, orient='split')

    # Comments Cyril made on the code
    test_items = test_items.head(1000)
    path = foldername3 + "/" + _id + "_test.json"
    convert_data_pldl_experiments(test_items, safety_label, 'Mindex', path)
    generate_sentence_embedding_only(test_items,foldername3,"test")



    # end changes from Cyril




    dfs_dev = dfs_combine[dfs_combine.isin(dev_items)]
    path = foldername3 + "/" + _id + "_dev.json"
    convert_data_pldl_experiments(dfs_dev, safety_label, 'Mindex', path)
    
    dfs_train = dfs_combine[dfs_combine.isin(train_items)]
    path = foldername3 + "/" + _id + "_train.json"
    convert_data_pldl_experiments(train_items, safety_label, 'Mindex', path)
    
    dfs_test = dfs_combine[dfs_combine.isin(test_items)]
    path = foldername3 + "/" + _id + "_test.json"
    convert_data_pldl_experiments(dfs_test, safety_label, 'Mindex', path)

    path = foldername1 + "/" + _id + "_annotations.json"
    ds_df = dfs_combine
    ds_df.to_json(path, orient='split')

    # dfs_dev = dfs_combine[dfs_combine.isin(dev_items)]
    # path = foldername3 + "/" + _id + "_dev.json"
    # convert_data_pldl_experiments(dfs_dev, safety_label, 'Mindex', path)
    #
    # dfs_train = dfs_combine[dfs_combine.isin(train_items)]
    # path = foldername3 + "/" + _id + "_train.json"
    # convert_data_pldl_experiments(dfs_train, safety_label, 'Mindex', path)
    #
    # dfs_test = dfs_combine[dfs_combine.isin(test_items)]
    # path = foldername3 + "/" + _id + "_test.json"
    # convert_data_pldl_experiments(dfs_test, safety_label, 'Mindex', path)


    X_train = pd.unique(dfs_train['query'])
    X_dev = pd.unique(dfs_dev['query'])
    X_test = pd.unique(dfs_test['query'])
    # annotators_array = np.full(len(annotators_parsed), -1)
    annotators_array = np.zeros(len(dfs_combine))

    generate_data_bert(dfs_train, foldername2, "train", label_dict, _id, X_train, annotators_array)
    generate_data_bert(dfs_test, foldername2, "test", label_dict, _id, X_test, annotators_array)
    generate_data_bert(dfs_dev, foldername2, "dev", label_dict, _id, X_dev, annotators_array)
    move_disco_embed_to_co(foldername2, foldername3)

    ##############################################
    labels = train_items  # .join(X_rows.set_index('message_id'),on='message_id')
    path = co_path + "prosocial_train.json"
    convert_data_pldl_experiments(labels, label_dict, 'message_id', path)

    labels = dev_items  # .join(X_rows.set_index('message_id'),on='message_id')
    path = co_path + "prosocial_dev.json"
    convert_data_pldl_experiments(labels, label_dict, 'message_id', path)

    labels = test_items  # .join(X_rows.set_index('message_id'),on='message_id')
    path = co_path + "prosocial_test.json"
    convert_data_pldl_experiments(labels, label_dict, 'message_id', path)

    move_disco_embed_to_co(folder, co_path)

def row_values_counter(colname, data_rows_pp):
    label_count_sub = Counter(data_rows_pp[colname])
    most_common, num_most_common = label_count_sub.most_common(1)[0]
    return most_common


#Function to calculate the labels majority
def process_prompts(my_dataset):
    data_items = pd.unique(my_dataset['safety_label'])
    processed_rows = pd.DataFrame()

    for prompt in tqdm(data_items):
        data_rows_prompt = my_dataset.loc[my_dataset['safety_label'] == prompt]
        data_row_df = data_rows_prompt.head(1)

        if data_rows_prompt.empty:
            # Handle empty rows if needed
            pdb.set_trace()

        data_row = data_row_df.to_dict()
        data_row['majority_safety_label'] = row_values_counter('safety_label', data_rows_prompt)
        data_row = pd.DataFrame(data_row)
        processed_rows = pd.concat([processed_rows, data_row], ignore_index = True)

    return processed_rows

def save_to_json(data, outputdir):
    if not os.path.exists(os.path.dirname(outputdir)):
        os.makedirs(os.path.dirname(outputdir))

    with open(outputdir, 'w') as outfile:
        outfile.write(json.dumps(data, indent=4))
        print("JSON saved to file" + outputdir)

# def convert_labels_hotencoding(data_items):
#     hotencoded = []
#
#     for index, row in data_items.iterrows():
#         labels = np.zeros(len(row['label_vector']))
#         labels[row['label_vector']] = 1
#         parsed_row = {}
#         parsed_row['item'] = row['Mindex']
#         # parsed_row['annotator'] = row['Aindex']
#         parsed_row['label'] = labels.astype(int)
#         hotencoded.append(parsed_row)
#
#     return pd.DataFrame(hotencoded)

def convert_labels_hotencoding(data_items, no_classes):
    hotencoded = []

    for index, row in data_items.iterrows():
        try:
            # Convert label_vector to integer if needed
            label_index = int(row['label_vector'])

            # Ensure label_index is within bounds
            if 0 <= label_index < no_classes:
                labels = np.zeros(no_classes)
                labels[label_index] = 1

                parsed_row = {
                    'label': labels.astype(int)  # Remove 'Mindex' and 'Aindex' from the parsed_row dictionary
                }

                hotencoded.append(parsed_row)
            else:
                print(f"Invalid label index: {label_index}")
        except ValueError:
            print(f"Invalid label_vector value: {row['label_vector']}")

    return pd.DataFrame(hotencoded)

def convert_labels_per_group(data_items,no_classes,grouping_category):
    encoded = []
    unique_data_items = pd.unique(data_items[grouping_category])
    for row in unique_data_items:
        encoded_row = {}
        labels = np.zeros(no_classes)
        items = data_items.loc[data_items[grouping_category] == row]
        for index,item in items.iterrows():
            labels[item['label_vector']]+=1
        encoded_row[grouping_category] = row
        encoded_row['label'] = labels.astype(int)
        encoded.append(encoded_row)
    return pd.DataFrame(encoded)

def read_splits(dev_input_file, train_input_file, test_input_file):
    data_dev = pd.read_csv(dev_input_file, sep="\t", header= None)
    dev_items = data_dev[2].tolist()
    data_train = pd.read_csv(train_input_file, sep="\t", header=None)
    train_items = data_train[2].tolist()
    data_test = pd.read_csv(test_input_file, sep="\t", header=None)
    test_items = data_test[2].tolist()

    return dev_items, train_items, test_items

# def unpivot(dframe, col_prompt, col_label):
#     df = dframe.melt(id_vars=[col_prompt], value_vars=[col_label])
#     df = df[df["value"] > 0]
#     df = df.drop(columns=["value"])
#     df = df.rename(columns={ 'variable': 'label', col_prompt: 'query'})
#     cols = ['query', 'label']
#     df = df[cols]
#
#     return df


# query,rots,safety_label,safety_annotations,safety_annotation_reasons,source
def csv_read(csvLocation,query,rots,safety_label,safety_annotations,safety_annotation_reasons,source):
    dframe = pd.read_csv(csvLocation, usecols = [query,rots,safety_label,safety_annotations,safety_annotation_reasons,source])
    cols = [query,rots,safety_label,safety_annotations,safety_annotation_reasons,source]
    dframe = dframe[cols]

    return dframe

if __name__ == "__main__":
    main()

