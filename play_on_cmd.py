# coding:utf-8
from q20 import *


feature = Q20_feature()

def one_play(nturn=20):
    turn = 0
    
    while True:
        choice = input("Please type 'start' to play a game.").lower()
        if choice == 'start':
            session = Q20_session(feature)
            break
    
    while turn < nturn:
        qid, question = session.get_question()
        choice = input(question).lower()
        
        if choice in ['y', 'yes']:
            session.append_qa(qid, 'yes')
            turn += 1
        elif choice in ['n', 'no']:
            session.append_qa(qid, 'no')
            turn += 1
        elif choice in ['ns', 'not sure']:
            session.append_qa(qid, 'ns')
            turn += 1
        else:
            print('type [y/yes]:[n/no]:[ns/not sure]')

    print("The role is {}".format(session.guessed_role()[1]))


if __name__ == '__main__':
    import sys
    args = sys.argv
    
    while True:
        one_play(int(args[1]))
        choice = input('continue? [yes/no]').lower()
        if choice in ['y', 'yes']:
            pass
        else:
            print("Thank you")
            break


        
