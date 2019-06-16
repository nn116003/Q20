# coding:utf-8
from q20 import *
import pandas as pd

def one_play(user, feature, nturn=20, random=False):
    user.init_role()
    session = Q20_session(feature)
    
    for turn in range(nturn):
        qid, qtext = session.get_question(random)
        ans = user.get_answer(qid)
        session.append_qa(qid, ans)

    rid, rname = session.guessed_role()
    session.commit(user.role)
    
    return rid == user.role


def one_life(N, nturn=10):
    feature = Q20_feature(sigma=0) # not use reference data = cold start
    user = Q20_player(feature.ref_table, feature.role_prior_prob())

    result = []
    for i in range(N):
        result.append(int(one_play(user, feature, nturn=nturn,)))

    return result


sample = 100
results = 0
life_range = 20000
nturn = 15


for i in range(sample):
    print(i)
    results += np.array(one_life(life_range, nturn=nturn))/float(sample)


np.save(open('./work/coldstart_1029.npy','wb'), results)
    
