# coding:utf-8
from q20 import *
import pandas as pd

feature = Q20_feature()
user = Q20_player(feature.ref_table, feature.role_prior_prob())

def one_play(nturn=20, random=False):
    user.init_role()
    session = Q20_session(feature)
    
    for turn in range(nturn):
        qid, qtext = session.get_question(random)
        ans = user.get_answer(qid)
        # print(session.feature.role_name(user.role), qtext, ans)
        session.append_qa(qid, ans)

    rid, rname = session.guessed_role()
    #print(rname, session.feature.role_name(user.role))
    
    return rid == user.role

N = 10000
acc = []
acc_r = []
for m in range(1, 28):
    correct = 0.
    correct_r = 0.
    for i in range(N):
        correct += int(one_play(m))
        correct_r += int(one_play(m, random=True))
    acc.append(correct/N)
    acc_r.append(correct_r/N)
#    print(m, correct/N, correct_r/N)

    
res = pd.DataFrame({'q20':acc, 'random':acc_r})
res.to_csv('work/acc_qs.csv', index=False)

