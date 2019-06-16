# coding:utf-8
import pandas as pd
import numpy as np
import os
import pickle

data_dir = "./data/zoo"

org_data = pd.read_csv(os.path.join(data_dir, "zoo.data"), header=None)

# create role table
role = pd.DataFrame({"rid":np.arange(org_data.shape[0]),
                     "name":org_data.iloc[:,0],
                     "rscore":1})
role.to_csv(os.path.join(data_dir, "role.csv"), index=False)

# create question table
q_text = [
    "毛が生えてる?",
    "羽はある?",
    "卵を産む?",
    "母乳で子供を育てる?",
    "飛べる?",
    "水生?",
    "捕食者?",
    "歯はある?",
    "背骨はある?",
    "肺呼吸?",
    "毒をもつ種がいる?",
    "ひれがある?",
    *["{} 本足?".format(s) for s in [0,2,4,5,6,8]],
    "しっぽがある?",
    "家畜?",
    "catsize?",
    *["哺乳類?","鳥?", "爬虫類", "魚類?", "両生類?",
      "虫?", "甲殻類/軟体動物?"]
]

question = pd.DataFrame({"qid":np.arange(len(q_text)),
                         "text":q_text,
                         "tag":None,
                         "value":None})
question.to_csv(os.path.join(data_dir, 'question.csv'), index=False)

# create rp table and reference question table
yes_arr = np.zeros((role.shape[0], len(q_text)))
yes_arr[:,:12] = org_data.values[:,1:13]
yes_arr[:,12:18] = pd.get_dummies(org_data.iloc[:, 13])
yes_arr[:,18:21] = org_data.values[:,14:17]
yes_arr[:,21:] = pd.get_dummies(org_data.iloc[:, 17])

ref_table = {'yes':yes_arr,
            'no':1-yes_arr}
#rp_table = {'yes':yes_arr,
#            'no':1-yes_arr,
#            'ns':np.zeros((role.shape[0], len(q_text)))}
rp_table = {'yes':np.zeros((role.shape[0], len(q_text))),
            'no':np.zeros((role.shape[0], len(q_text))),
            'ns':np.zeros((role.shape[0], len(q_text)))}

pickle.dump(ref_table,
            open(os.path.join(data_dir, 'ref_table.pkl'), 'wb'))
pickle.dump(rp_table,
            open(os.path.join(data_dir, 'rp_table.pkl'), 'wb'))

# create mutex table
yy = rp_table['yes'].transpose() @ rp_table['yes']
yn = rp_table['yes'].transpose() @ rp_table['no']
ny = rp_table['no'].transpose() @ rp_table['yes']
nn = rp_table['no'].transpose() @ rp_table['no']
ns = np.zeros((len(q_text), len(q_text)))

mutex_table = {'yy':yy/(yy+yn+1e-10), 'yn':yn/(yy+yn+1e-10),
               'ny':ny/(ny+nn+1e-10), 'nn':nn/(ny+nn+1e-10),
               'ns':ns}
pickle.dump(mutex_table,
            open(os.path.join(data_dir, 'mutex_table.pkl'), 'wb'))

