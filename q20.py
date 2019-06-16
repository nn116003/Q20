# coding:utf-8
import os
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import utils

np.random.seed(0)

class Q20_feature():
    def __init__(self, data_dir=Path("./data/zoo"),
                 ckpt_dir=Path("./ckpts/zoo"), sigma=1000, alpha=1, beta=1):
        self.load_features(data_dir)
        self.ckpt_dir = ckpt_dir
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta

        self.set_yes_prob()
        self.ans_neg_entropy = -utils.entropy(self.yes_prob)

    def role_name(self, row):
        return self.role.name[row]

    def question_text(self, row):
        return self.question.text[row]

    def role_prior_prob(self):
        return self.role.rscore.values.astype(float)/self.role.rscore.values.sum()
        

    def load_features(self, data_dir):
        self.role = pd.read_csv(data_dir.joinpath('role.csv'))

        self.question = pd.read_csv(data_dir.joinpath('question.csv'))

        self.mutex = pickle.load(
            open(data_dir.joinpath('mutex_table.pkl'), 'rb')
        )
        # row: role
        # col: question
        self.ref_table = pickle.load(
            open(data_dir.joinpath('ref_table.pkl'), 'rb')
        )
        self.rp_table = pickle.load(
            open(data_dir.joinpath('rp_table.pkl'), 'rb')
        )
        
    def set_yes_prob(self):
        self.yes_prob = (self.rp_table['yes'] + self.sigma * self.ref_table['yes']) \
                   / (self.rp_table['yes'] + self.rp_table['no']
                      + self.alpha + self.sigma*self.ref_table['yes'])


    def qa_prob(self, qid, ans):
        tmp1 = self.rp_table[ans][:, qid] + \
               self.sigma*self.ref_table[ans][:, qid]
        tmp2 = self.rp_table['yes'][:, qid] + self.rp_table['no'][:, qid] + \
               self.rp_table['ns'][:, qid] + self.beta + self.sigma*self.ref_table[ans][:, qid]

        return tmp1 / tmp2

    def mutex_fix_score(self, qid, ans):
        if ans == 'yes':
            fix_score = self.mutex['yy'][qid,:] + self.mutex['yn'][qid,:]
        elif ans == 'no':
            fix_score = self.mutex['ny'][qid,:] + self.mutex['nn'][qid,:]
        else:
            fix_score = self.mutex['ns'][qid,:]

        return fix_score

    def update(self, qa_list, ans_role, update_mutex=True):
        for qa in qa_list:
            qid = qa['q']
            aid = qa['a']
            self.rp_table[aid][ans_role, qid] += 1
        self.set_yes_prob()


    def update_mutex(self):
        yy = self.rp_table['yes'].transpose() @ self.rp_table['yes']
        yn = self.rp_table['yes'].transpose() @ self.rp_table['no']
        ny = self.rp_table['no'].transpose() @ self.rp_table['yes']
        nn = self.rp_table['no'].transpose() @ self.rp_table['no']
        ns = np.zeros((len(q_text), len(q_text)))

        mutex_table = {'yy':yy/(yy+yn+1e-10), 'yn':yn/(yy+yn+1e-10),
                       'ny':ny/(ny+nn+1e-10), 'nn':nn/(ny+nn+1e-10),
                       'ns':ns}

    def save(self):
        ckpt_dir = self.ckpt_dir
        self.role.to_csv(ckpt_dir.joinpath('role.csv'),
                         index=False)
        self.question.to_csv(ckpt_dir.joinpath('question.csv'),
                             index=False)
        pickle.dump(self.mutex,
                    open(ckpt_dir.joinpath('mutex_table.pkl'), 'wb')
        )
        pickle.dump(self.ref_table,
                    open(ckpt_dir.joinpath('ref_table.pkl'), 'wb')
        )
        pickle.dump(self.rp_table,
                    open(ckpt_dir.joinpath('rp_table.pkl'), 'wb')
        )
        
        

class Q20_session():
    def __init__(self, feature, lmda=0.):
        self.feature = feature
        self.lmda = lmda
        self.role_prob = feature.role_prior_prob()
        self.qa_list = []
        self.mutex_info = np.zeros(self.feature.question.shape[0])

        self.qscore = 0
        self._update_qscore()
        

    def _update_qscore(self):
        # [q, r] * [r,1]
        e_ane = (self.feature.ans_neg_entropy.transpose() @ self.role_prob.reshape(-1,1)).reshape(-1)
        e_yp = (self.feature.yes_prob.transpose() @ self.role_prob.reshape(-1,1)).reshape(-1)
        e_yp_entropy = utils.entropy(e_yp)

        if len(self.qa_list) > 0:
            qid = self.qa_list[-1]['q']
            ans = self.qa_list[-1]['a']
            self.mutex_info[qid] -= 100000
            fix_score = self.feature.mutex_fix_score(qid, ans)
            self.mutex_info -= self.lmda * fix_score
            
        self.qscore = e_ane + e_yp_entropy + self.mutex_info

    def _update_role_prob(self):
        ans = self.qa_list[-1]['a']
        qid = self.qa_list[-1]['q']
        
        if ans in ['yes', 'no']:
            prod_factor = self.feature.qa_prob(qid, ans)
        else:
            prod_factor = 1 - self.feature.qa_prob(qid, 'yes') \
                          - self.feature.qa_prob(qid, 'no')

        self.role_prob *= prod_factor

    def append_qa(self, q, a):
        self.qa_list.append({'q':q, 'a':a})
        self._update_role_prob()
        self._update_qscore()

    def guessed_role(self):
        idx = self.role_prob.argmax()
        name = self.feature.role_name(idx)
        return idx, name

    def get_question(self, random=False):
        if random:
            qs = [qa['q'] for qa in self.qa_list]
            l = len(self.qscore)
            candidates = set(list(np.arange(l))) - set(qs)
            idx = np.random.choice(list(candidates))
        else:
            #tmp = self.qscore + self.qscore.min()
            #tmp = np.exp(self.qscore)
            #q_prob = tmp/tmp.sum()
            #idx = np.random.choice(len(self.qscore), p=q_prob)
            
            #idx = np.random.choice(self.qscore.argsort()[-3:])
            idx = self.qscore.argmax()
            
        text = self.feature.question_text(idx)
        return idx, text

    def commit(self, ans_role):
        self.feature.update(self.qa_list, ans_role)
        

    
class Q20_player():
    def __init__(self, ref_table, role_prob, noize=0):
        self.ref_table = ref_table
        self.role_prob = role_prob
        self.noize = noize
        self.role_num = len(role_prob)
        self.q_num = ref_table['yes'].shape[1]
        self.init_role()

    def init_role(self):
        self.role = self.sample_role()
        ans_list = np.array(['ns']*self.q_num).astype('<U3')
        is_yes = self.ref_table['yes'][self.role, :].astype(bool)
        is_no = self.ref_table['no'][self.role, :].astype(bool)
        ans_list[is_yes] = 'yes'
        ans_list[is_no] = 'no'
        self.ans_list = ans_list

    def sample_role(self):
        return np.random.choice(self.role_num, p=self.role_prob)

    def get_answer(self, q_id, random=False):
        correct_ans = self.ans_list[q_id]
        if random:
            r = np.random.uniform()
            if correct_ans == 'yes':
                return np.random.choice(['no', 'ns']) if r < self.noize else 'yes'
            elif correct_ans == 'no':
                return np.random.choice(['yes', 'ns']) if r < self.noize else 'no'
            else:
                return 'ns'
        else:
            return correct_ans
        





















