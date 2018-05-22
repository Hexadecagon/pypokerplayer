from Processing import get_bot_info
from Raise_Sizes import generate_raise_sizes

def optimal_raise_Qvalues(valid_actions,hole_card,round_state,network,player_index,target_mode,features=None): #compute Qvalues for the optimal raise size.
    if features == None: 
        observation = get_bot_info(valid_actions,hole_card,round_state,player_index)
    else:
        observation = features
        
    def get_Q(): #get Q values given a particular raise size (3rd element is for raise)
        return network.get_qs(observation, target_mode).tolist()[0]

    all_in = valid_actions[2]["amount"]["max"] #constants
    min_bet = valid_actions[2]["amount"]["min"]
    pot = round_state["pot"]["main"]["amount"]

    if(min_bet == -1 and all_in == -1):
        return network.get_qs(observation, target_mode).tolist()[0][0:2], -1
    
    raise_sizes = generate_raise_sizes(min_bet,all_in)
   # print(raise_sizes)    
    Qs = get_Q()
    
    max_raise_Q = max(Qs)   #select best qvalue
    
    max_raise_Q_index = Qs.index(max_raise_Q) #find raise size corresponding to best qvalue

    if max_raise_Q_index == 0:
        optimal_raise_size = 0
    elif max_raise_Q_index == 1:
        optimal_raise_size = valid_actions[1]["amount"]
    else:
        optimal_raise_size = raise_sizes[max_raise_Q_index-2]


    return Qs, optimal_raise_size
