import pandas as pd
import numpy as np
import json


def squad_to_dataframe(data_path, record_path=['data', 'paragraphs', 'qas',
                     'answers']):

    # record_path: path to deepest level in json file

    #
    jsonData = json.loads(open(data_path).read())

    # parsing different levels
    dfDeep = pd.json_normalize(jsonData, record_path)
    dfDeepMinus1 = pd.json_normalize(jsonData, record_path[:-1])
    dfDeepMinus2 = pd.json_normalize(jsonData, record_path[:-2])

    # concatinating into single dataframe
    contex = np.repeat(dfDeepMinus2['context'].values, dfDeepMinus2.qas.str.len())
    qid = np.repeat(dfDeepMinus1['id'].values, dfDeepMinus1['answers'].str.len())
    dfDeepMinus1['context'] = contex
    dfDeep['q_idx'] = qid
    finalDf = pd.concat([dfDeepMinus1[['id', 'question', 'context']].set_index('id'), dfDeep.set_index('q_idx')], 1,
                     sort=False).reset_index()
    finalDf['c_id'] = finalDf['context'].factorize()[0]

    return finalDf



# training data
dataPath = 'Data/train-v2.0.json'
recordpath = ['data', 'paragraphs', 'qas', 'answers']
df = squad_to_dataframe(data_path=dataPath, record_path=recordpath)


df.columns
df.head()