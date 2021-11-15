import torch
import datetime
import numpy as np
import joblib

def tprint(s):
    '''
        print datetime and s
        @params:
            s (str): the string to be printed
    '''
    print('{}: {}'.format(
        datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'), s),
          flush=True)

def to_tensor(data, cuda, exclude_keys=[]):
    '''
        Convert all values in the data into torch.tensor
    '''
    for key in data.keys():
        if key in exclude_keys:
            continue
        if isinstance(data[key], list) and not isinstance(data[key][0],int):
            tmp = []
            for item in data[key]:
                if cuda != -1:
                    tmp.append(torch.from_numpy(item).cuda(cuda))
                else:
                    tmp.append(torch.from_numpy(item))
            data[key] = tmp
        else:
            data[key] = torch.from_numpy(data[key])
            if cuda != -1:
                data[key] = data[key].cuda(cuda)

    return data


def select_subset(old_data, new_data, keys, idx, index2vector, max_len=None, add_label_vector=False):
    '''
        modifies new_data

        @param old_data target dict
        @param new_data source dict
        @param keys list of keys to transfer
        @param idx list of indices to select
        @param max_len (optional) select first max_len entries along dim 1
    '''
    for k in keys:
        if isinstance(old_data[k],list) and not isinstance(old_data[k][0],int):
            tmp = []
            for i in idx:
                if max_len is not None:
                    tmp.append(np.array(old_data[k][i][:max_len]))
                else: 
                    tmp.append(np.array(old_data[k][i]))
            if max_len is not None:
                tmp.append(np.array(old_data[k][i][:max_len]))
            else: 
                tmp.append(np.array(old_data[k][i]))
            new_data[k] = tmp
        else:
            new_data[k] = old_data[k][idx]
            if max_len is not None and len(new_data[k].shape) > 1:
                new_data[k] = new_data[k][:,:max_len]
    
    # if add_label_vector:
    label_vectors = np.zeros((len(new_data['label']),300),dtype=np.float)
    unique_label_vectors = []
    unique_label_index = set()
    for i,label_index in enumerate(new_data['label']):
        label_vectors[i] = index2vector[label_index]
        if label_index not in unique_label_index:
            unique_label_index.add(label_index)
            unique_label_vectors.append(index2vector[label_index])
    new_data['label_vectors'] = label_vectors

    unique_label_vectors = np.vstack(unique_label_vectors)
    new_data['unique_label_vectors'] = unique_label_vectors
        
    return new_data
