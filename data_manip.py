import pandas as pd
import random
import os
import csv

seed = 42
random.seed(42)


def remove_labeled(fn, fn_destination):
    '''
    Removes all labeled data
    '''
    data_df = pd.read_csv(fn)
    
    drop_idx = data_df[(data_df['toxic']==0) & (data_df['severe_toxic']==0) & (data_df['obscene']==0)
                  & (data_df['threat']==0) & (data_df['insult']==0) & (data_df['identity_hate']==0)].index
    data_df.drop(drop_idx,inplace=True)

    data_df.to_csv(fn_destination,index=False)
    return data_df

def generate_data(fn,fn_destination):
    '''
    Generate new data by shuffling each line in input file
    '''
    new_data_csv = os.path.join(os.getcwd(), fn_destination)

    data = pd.read_csv(fn)

    with open(new_data_csv, 'a', encoding='utf-8') as new_data_file:
        labels = ['id','comment_text','toxic','severe_toxic','obscene','threat',
                  'insult','identity_hate']
        row_str = ','.join(map(str, labels)) + '\n'
        row_str = 'id,comment_text,toxic,severe_toxic,obscene,threat,insult,identity_hate\n'
        new_data_file.write(row_str)
        for index, row in data.iterrows():
            text = row['comment_text'].split()
            new_text = ' '.join(random.sample(text,len(text)))
            row['comment_text'] = new_text
            row_str = ','.join(map(str, row.values.tolist())) + '\n'
            new_data_file.write(row_str)


def add_gen_to_data(data_fn, gen_data_fn, destination_df):
    '''
    Concats data two files
    Saves to csv
    '''
    data = pd.read_csv(data_fn)
    gen_data = pd.read_csv(gen_data_fn)
    
    new_data = pd.concat([data,gen_data])

    new_data.to_csv(destination_df,index=False)
    return new_data

def row_not_classified(row):
    '''
    Returns true if row is unlabeled
    Returns false if row has any label, including multiple labels
    '''
    # Check if all columns except 'id' and 'comment_text' are "0"
    return all(value == "0" for key, value in row.items() if key not in ['id', 'comment_text'])

def prune_unclassified_data(input_file, output_file, remove_percentage):
    '''
    Prunes input file
    Unlabeled rows are pruned with probability remove_percentage
    '''
    with open(input_file, 'r') as f_in, open(output_file, 'w', newline='') as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.writer(f_out)
        
        header = ['id', 'comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        writer.writerow(header)
        for row in reader:
            if row['comment_text'].strip() != '':  # Check if 'comment_text' is not an empty string
                if row_not_classified(row):
                    if random.random() > remove_percentage:  # Check if the random value is above the percentage
                        writer.writerow([row['id'], row['comment_text'], row['toxic'], row['severe_toxic'], row['obscene'], row['threat'], row['insult'], row['identity_hate']])
                else:
                    writer.writerow([row['id'], row['comment_text'], row['toxic'], row['severe_toxic'], row['obscene'], row['threat'], row['insult'], row['identity_hate']])


# original data
orig_data= os.path.join(os.getcwd(), 'data\\train_cleaned.csv')

# remove labels
no_label = os.path.join(os.getcwd(), 'data\\train_cleaned_pruned.csv')
no_label_df = remove_labeled(orig_data,no_label)

# generate from labeled data
generated = os.path.join(os.getcwd(), 'data\\generated.csv')
generate_data_df = generate_data(no_label,generated)

# prune original data
percents = [50, 60, 70, 80, 90]
for percent in percents:
    pruned = 'data\\' + str(percent) + 'pruned.csv'
    pruned = os.path.join(os.getcwd(), pruned)
    prune_unclassified_data(orig_data, pruned, percent/100)

# combine generated data with pruned data
for percent in percents:
    path1 = 'data\\' + str(percent) + 'pruned.csv'
    data1_datapath = os.path.join(os.getcwd(), path1)
    data2_datapath = os.path.join(os.getcwd(), 'data\\generated.csv')
    dest_path = 'data\\' + str(percent) + 'pruned_w_gen.csv'
    destination = os.path.join(os.getcwd(), dest_path)
    add_gen_to_data(data1_datapath,data2_datapath,destination)

