import random
import math
import numpy as np
from Raise_Sizes import generate_raise_size,sizes
from Optimal_Raise_Qvalues import optimal_raise_Qvalues

def softmax_new(x,t):
    """Compute softmax values for each sets of scores in x."""
    new_nums = []
    for index,num in enumerate(x):
        if index == 0 or index == 1:
            new_nums.append(math.exp(num/t))
        else:
            if index-1 < len(sizes):
                jump =sizes[index-1] -sizes[index-2]
            else:
                jump = 0.08
            new_nums.append(math.exp(num/t)*jump*2)
    
    s = sum(new_nums)
    return [n/s for n in new_nums]

def choose_bayesian(valid_actions,hole_card,round_state,network,epsilon,player_index,features=None,verbose=False):
    Qvalues,raise_size = optimal_raise_Qvalues(valid_actions,hole_card,round_state,network,player_index,"original",features)
    
    action_probs = softmax_new(Qvalues,100*epsilon)
    
    if  verbose:
        print(Qvalues,raise_size)
        print(action_probs)
    choices = list(range(len(action_probs)))
    choice = choices.index(np.random.choice(choices,p=action_probs))
    
    if choice == 0:
        if valid_actions[1]["amount"] == 0:
            return "call", 0 
        return "fold", 0
    elif choice == 1:
        return "call", valid_actions[1]["amount"]
    elif choice >= 2:
        min_bet = valid_actions[2]["amount"]["min"]
        all_in = valid_actions[2]["amount"]["max"]
        return "raise",generate_raise_size(min_bet,all_in,choice-2) 
    
    
