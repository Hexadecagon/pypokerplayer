from Processing import get_bot_info
import random
def optimal_raise_Qvalues(valid_actions,hole_card,round_state,network,player_index,target_mode,features=None): #compute Qvalues for the optimal raise size.
    if features == None: 
        observation = get_bot_info(valid_actions,hole_card,round_state,player_index)
    else:
        observation = features
        
    def get_Q(raise_size): #get Q values given a particular raise size (3rd element is for raise)
        return network.get_qs(observation + [raise_size], target_mode).tolist()[0][2]

    all_in = valid_actions[2]["amount"]["max"] #constants
    min_bet = valid_actions[2]["amount"]["min"]
    pot = round_state["pot"]["main"]["amount"]

    if(min_bet == -1 and all_in == -1):
        return network.get_qs(observation + [-1], target_mode).tolist()[0][0:2], -1
    
    raise_sizes = [random.uniform(min_bet,all_in) for x in range(5)] #various raise sizes and Q values for such
    Qs = [get_Q(raise_size) for raise_size in raise_sizes]
            
    increment = int(pot/5)

    for _ in range(10): #increase for more accuracy
        increment *= 0.9 #how fast convergence is
        for index, r in enumerate(raise_sizes):

            this_r = Qs[index] #current Qvalue
            higher_bet = min(all_in, r+increment)
            lower_bet = max(min_bet, r-increment)

                
            higher_r =get_Q(higher_bet)#grab q values
            lower_r = get_Q(lower_bet)
            
            if(this_r > higher_r and this_r > lower_r):
                pass
            elif higher_r > lower_r: #if higher raise is better, update raise to be higher
                raise_sizes[index] = higher_bet
                Qs[index] = higher_r
                
            elif lower_r>higher_r: #lower raise is better
                raise_sizes[index] = lower_bet
                Qs[index] = lower_r
    
    max_raise_Q = max(Qs)   #select best qvalue
    
    max_raise_Q_index = Qs.index(max_raise_Q) #find raise size corresponding to best qvalue

    optimal_raise_size = raise_sizes[max_raise_Q_index]

    final_Q = network.get_qs(observation + [optimal_raise_size],target_mode).tolist()[0] 
    return final_Q, optimal_raise_size
