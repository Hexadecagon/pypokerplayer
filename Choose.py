import random
import math
from Optimal_Raise_Qvalues import optimal_raise_Qvalues
def choose_action(valid_actions,hole_card,round_state,network, epsilon,player_index,features = None,verbose = False):
    if random.random() > epsilon:
        Qvalues,raise_size =optimal_raise_Qvalues(valid_actions,hole_card,round_state,network,player_index,"original",features)
        if  verbose:
            print(Qvalues,raise_size)
        max_Q = max(Qvalues)
            
        if Qvalues.index(max_Q) == 0:
            if valid_actions[1]["amount"] == 0:
                return "call", 0 
            return "fold", 0
        elif Qvalues.index(max_Q) == 1:
            return "call", valid_actions[1]["amount"]
        elif Qvalues.index(max_Q) == 2:
            return "raise", raise_size
    else: #choose a random action.
        if  verbose:
            print("random")
        new_random = random.random()
        if new_random < 0.25:
            return "fold",0
        elif new_random< 0.50:
            return  "call", valid_actions[1]["amount"]
        else:
            if valid_actions[2]["amount"]["min"] == -1:
                return  "call", valid_actions[1]["amount"]
            return "raise", random.uniform(valid_actions[2]["amount"]["min"], min(valid_actions[2]["amount"]["max"],round_state["pot"]["main"]["amount"]))


