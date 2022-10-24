import csv

import pandas as pd
import numpy as np
from numpy import log2 as log
import pprint

# PREPROCESSING
attribute_table_file = open('attribute_table_output.txt', 'w', encoding='utf-8')
file = open('hw1.train.col', 'r', encoding='utf-8')
writer = csv.writer(attribute_table_file)

top_words = [',', 'on', '<s>', 'about', 'the', 'of', 'and', 'asked', 'decide', 'over', 'is', 'determine', '-',
             'unclear', 'to', 'say', 'know', 'clear', 'national', '"', 'see', 'questioned', 'bad', 'into', 'extreme',
             'severe', 'investigating', 'or', 'question', 'deciding', 'cold', 'but', 'out', 'wonder', 'consider',
             'sure', 'warm', 'at', 'was', 'hot', 'considering', 'wet', 'seen', 'winter', 'wondered', 'wondering',
             'in', 'known', 'as', 'asking','the', 'it', 'to', 'he', 'they', 'or', 'you', 'a', ',', '</s>', 'that',
             'there', 'service', 'she', 'we', 'this', 'and', 'in', 'conditions', 'is', 'any', 'his', 'i', '-', 'their',
             'trump', 'mr', 'an', 'these', 'has', 'on', 'as', 'people', '"', 'those', 'events', 'forecast', 'mr.',
             'was', 'for', 'will', 'warning', 'by', 'such', 'its', ':', 'her', 'channel', 'anyone', 'at']
header =['label']
for word in top_words:
    header.append(word)
writer.writerow(header)

for line in file:
    line = line.strip()
    attr_Array = line.split(' ')
    label = attr_Array[0]
    target_index = int(attr_Array[1])
    sentence_array = attr_Array[2:]
    if target_index < 0 or target_index >= len(sentence_array):
        pass
    else:
        row = [label]
        for word in top_words:
            position = 'None'
            if target_index != 0 and sentence_array[target_index - 1] == word:
                position = '1Before'
            if target_index != len(sentence_array) - 1 and sentence_array[target_index + 1] == word:
                position = '1After'
            row.append(position)
        writer.writerow(row)

attribute_table_file.close()


def get_entropy(df):
    Class = 'label'
    entropy = 0
    values = df[Class].unique()
    for value in values:
        fraction = df[Class].value_counts()[value] / len(df[Class])
        entropy += -fraction * np.log2(fraction)
    return entropy


def get_entropy_attribute(df, attribute):
    Class = 'label'  # To make the code generic, changing target variable class name
    target_variables = df[Class].unique()  # This gives all 'Yes' and 'No'
    variables = df[attribute].unique()  # This gives different features in that attribute (like 'Hot','Cold' in Temperature)
    entropy_tot = 0
    for variable in variables:
        entropy = 0
        for target_variable in target_variables:
            num = len(df[attribute][df[attribute] == variable][df[Class] == target_variable])
            den = len(df[attribute][df[attribute] == variable])
            fraction = num / (den + eps)
            entropy += -fraction * log(fraction + eps)
        fraction2 = den / len(df)
        entropy_tot += -fraction2 * entropy
    return abs(entropy_tot)


def get_best_info_gain(df):
    entropy_att = []
    info_gain = []
    # key iterates between df.keys() aka column names
    for key in df.keys()[1:]:
        entropy_att.append(get_entropy_attribute(df, key))
        info_gain.append(get_entropy(df) - get_entropy_attribute(df, key))
    return df.keys()[1:][np.argmax(info_gain)]


def get_sub_table(df, node_attr, value):
    return df[df[node_attr] == value].reset_index(drop=True)


def build_tree(df, max_depth, tree=None, curr_lvl=0):
    Class = 'label'
    node = get_best_info_gain(df)
    att_value = np.unique(df[node])
    if tree is None:
        tree = {}
        tree[node] = {}

    for value in att_value:
        sub_table = get_sub_table(df, node, value)
        values_counts = sub_table[Class].value_counts()
        if len(values_counts) == 1 or curr_lvl >= max_depth:
            tree[node][value] = values_counts.idxmax()
        else:
            tree[node][value] = build_tree(sub_table, max_depth, None, curr_lvl + 1)
    return tree


eps = np.finfo(float).eps

df = pd.read_csv("attribute_table_output.txt")

df = df.sample(frac=0.1)
print(df)

tree = build_tree(df, 5)
pprint.pprint(tree)

print(df.keys())
print(get_entropy(df))
# for each in df.keys()[1:]:
#    print(get_entropy_attribute(df, each))


# Testing
def test_data(test, tree, default=None):
    attribute = next(iter(tree))
    #print(attribute)
    if test[attribute] in tree[attribute].keys():
        #print(tree[attribute].keys())
        #print(test[attribute])
        result = tree[attribute][test[attribute]]
        if isinstance(result, dict):
            return test_data(test, result)
        else:
            return result
    else:
        return default

test_file = open('hw1.test.col', 'r', encoding='utf-8')
tests = []

for line in test_file:
    line = line.strip()
    attr_Array = line.split(' ')
    label = attr_Array[0]
    target_index = int(attr_Array[1])
    sentence_array = attr_Array[2:]
    if target_index < 0 or target_index >= len(sentence_array):
        pass
    else:
        curr_test = {'label': label}
        for word in top_words:
            position = 'None'
            if target_index != 0 and sentence_array[target_index - 1] == word:
                position = '1Before'
            if target_index != len(sentence_array) - 1 and sentence_array[target_index + 1] == word:
                position = '1After'
            curr_test[word] = position
        tests.append(curr_test)

count_true = 0
count_false = 0
for test in tests:
    cat = (test_data(test, tree))
    result = cat == test['label']
    print(result)
    if result == True:
        count_true +=1
    else:
        count_false +=1
print(count_true / (count_false + count_true))



#ans = test_data(test, tree)
#print(ans)
