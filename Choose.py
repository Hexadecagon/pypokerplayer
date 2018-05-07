import random
import math
from Optimal_Raise_Qvalues import optimal_raise_Qvalues
def choose_action(valid_actions,hole_card,round_state,network, epsilon,player_index,features = None):
    if random.random() > epsilon:
        Qvalues,raise_size =optimal_raise_Qvalues(valid_actions,hole_card,round_state,network,player_index,"original",features)
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
        new_random = random.random()
        print("random")
        if new_random < 0.25:
            return "fold",0
        elif new_random< 0.50:
            return  "call", valid_actions[1]["amount"]
        else:
            if valid_actions[2]["amount"]["min"] == -1:
                return  "call", valid_actions[1]["amount"]
            return "raise", random.uniform(valid_actions[2]["amount"]["min"], min(valid_actions[2]["amount"]["max"],round_state["pot"]["main"]["amount"]))

def choose_action_boltzmann(valid_actions,hole_card,round_state,network,player_index,is_preflop):
    def get_Q(raise_size): 
        return network.get_qs(observation + [raise_size], target_mode).tolist()[0][2]
    def softmax(x):
        exponentiated = [math.exp(y) for y in x]
        s = sum(exponentiated)
        return [z/s for z in exponentiated]
    def redistribute(x):
        redistributed = [z + 0.2 for z in x]
        s = sum(redistributed)
        return [z/s for z in redistributed]
    def random_pick(probabilities):
        new_random = random.random()
        total = 0
        for index,prob in enumerate(probabilities):
            total += prob
            if new_random < total:
                return index
            
    Qvalues,raise_size =optimal_raise_Qvalues(valid_actions,hole_card,round_state,network,player_index,"original",is_preflop)
    print(Qvalues,raise_size)
    certainties = redistribute(softmax(Qvalues))
    print(certainties)
    choice = random_pick(certainties)
    if choice == 0:
        return "fold", 0
    elif choice:
        return  "call", valid_actions[1]["amount"]
    else:
        min_bet = valid_actions[2]["amount"]["min"]
        max_bet = valid_actions[2]["amount"]["max"]
        diff = max_bet - min_bet
        qs = [get_Q(bet)  for bet in [n*diff/10 for n in range(10)]]
        return "raise", qs[random_pick(softmax(qs))]
