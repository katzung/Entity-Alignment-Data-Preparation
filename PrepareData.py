import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from utils import *
from AlignTypes import align_types


def dimension_decrease(X):
    tsne = TSNE(n_components=2, init='pca', random_state=123)
    X_decrease = tsne.fit_transform(X)
    return X_decrease


def form_pairs(source, results):
    links = dict()
    ent_links = source + 'ent_links'
    with open(ent_links, encoding='utf-8') as f:
        for line in f:
            th = line[:-1].split('\t')
            links[th[0]] = th[1]
                                  
    pairs = dict()
    kgs_ids = get_kgs_ids(results, inverse=True)
    for uri in links.keys():
        id1 = kgs_ids[uri]
        uri2 = links[uri]
        id2 = kgs_ids[uri2]
        pairs[id1] = id2
        pairs[id2] = id1
        
    return pairs


def determine_lang(ids):
    even_list = (ids % 2) == 0
    lang_list = []
    for elem in even_list:
        if elem == True:
            lang_list.append('English')
        else:
            lang_list.append('Russian')
    return lang_list


def get_names(source, targets):
    names = list()
    for elem in targets:
        name = source[elem]
        #name = delete_host(name)
        names.append(name)
    return names

def get_names_and_delete_hosts(source, targets):
    names = list()
    for elem in targets:
        name = source[elem]
        name = delete_host(name)
        names.append(name)
    return names

def dict_types(filepath):
    types = {}
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            line = line.replace('<', '')
            line = line.replace('>', '')
            line = line.strip().split(' ')
            if len(line) == 4:
                #name = delete_host(line[0])
                name = line[0]
                type_ = delete_host(line[2])
                types[name] = type_
    return types


def double_dict(fpath_1, fpath_2):
    types_1 = dict_types(fpath_1)
    types_2 = dict_types(fpath_2)

    full_types = dict()
    full_types.update(types_2)
    full_types.update(types_1)

    return full_types


def prepare_data(source, results, ftp_1, ftp_2):
    embeds = np.load(results + 'ent_embeds.npy')

    print('Dimension decrease')
    embeds = dimension_decrease(embeds)
    prepared_data = pd.DataFrame(embeds, columns=['x', 'y'])
    
    print('Pairs definition')
    ids_1 = list(prepared_data.index)
    prepared_data['Ent1_ID'] = ids_1
    ref_pairs = form_pairs(source, results)
    prepared_data['Ent2_ID'] = [ref_pairs[x] for x in ids_1]

    print('Name definition')
    kgs_ids = get_kgs_ids(results)
    prepared_data['Ent1'] = get_names(kgs_ids, prepared_data['Ent1_ID'])
    prepared_data['Ent2'] = get_names(kgs_ids, prepared_data['Ent2_ID'])

    print('Language definition')
    prepared_data['Language'] = determine_lang(prepared_data['Ent1_ID'])

    print('Types definition')
    ent_types = double_dict(ftp_1, ftp_2)
    prepared_data['Type'] = prepared_data['Ent1'].map(ent_types)

    print('Types alignment')
    prepared_data = align_types(prepared_data)

    print('Delete hosts')
    kgs_ids = get_kgs_ids(results)
    prepared_data['Ent1'] = get_names_and_delete_hosts(kgs_ids, prepared_data['Ent1_ID'])
    prepared_data['Ent2'] = get_names_and_delete_hosts(kgs_ids, prepared_data['Ent2_ID'])

    return prepared_data


if __name__ == '__main__':
    instanceTypesEn = 'data/instance_types/instance_types_en.ttl'
    instanceTypesRu = 'data/instance_types/instance_types_ru.ttl'

    dbp15kPath = 'data/EN_RU_15K_V1/'

    alignmentResultsPath = 'data/MultiKE/EN_RU_15K_V1/631/20240416192852/'
    outputFilename = 'MultiKE_EN_RU_15K_V1'

    df = prepare_data(dbp15kPath, alignmentResultsPath, instanceTypesEn, instanceTypesRu)

    print('Saving csv')

    resultFolder = 'output'
    path = resultFolder + '/' + outputFilename + '.csv'
    df.to_csv(path, index=False)