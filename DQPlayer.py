from DQObjects import Event
from pypokerengine.players import BasePokerPlayer
from Choose import choose_action
from Processing import get_bot_info
import random

class DQPlayer(BasePokerPlayer):
  def __init__(self,network,self_network,buffer,bufferSL,number,starting_parameters=None,verbose=False):
      BasePokerPlayer.__init__(self)
      self.player_index=number
      self.iterations = 0

      if starting_parameters == None:
        self.epsilon = 0.5
        self.anticipatory = 0.8 
        self.min_epsilon = 0.05
        self.update_num = 100
        self.decay = 0.00025
      else:
        self.epsilon = starting_parameters[0]
        self.anticipatory = starting_parameters[1]
        self.decay = starting_parameters[2]
        self.min_epsilon = starting_parameters[3]
        self.update_num = starting_parameters[4]

      
      self.buffer = buffer
      self.bufferSL = bufferSL
      
      self.last_action = None
      self.last_state = None
      self.last_features = None
      self.last_chose_greedy = None
      
      self.verbose = verbose

      self.network = network
      self.self_network = self_network

  def compute_reward(self,round_state):

    last_stack = self.last_state[2]["seats"][self.player_index]["stack"] #might not work perfectly. Might need to set maxQ to 0, that would make sense. 
    new_stack = round_state["seats"][self.player_index]["stack"]
    reward = new_stack - last_stack
    return reward

  def declare_action(self, valid_actions, hole_card, round_state):
    verbose = self.verbose
    features = get_bot_info(valid_actions,hole_card,round_state,self.player_index)
    self.run_updates(valid_actions,hole_card,round_state,features)

    if  verbose:
      print(hole_card)
      
    if random.random() < self.anticipatory:
        #choose epsilon-greedy action
        action = choose_action(valid_actions,hole_card,round_state,self.network,self.epsilon,self.player_index,features,verbose)
        self.last_chose_greedy = True
    else:
        action = self.self_network.predict([valid_actions,hole_card,round_state],self.player_index,features)
        self.last_chose_greedy = False
        if verbose:
          print("predicted")
        
    #update epsilon,last state, last action
    self.epsilon =max(self.min_epsilon,self.epsilon-self.decay)    
    self.last_state = [valid_actions,hole_card,round_state]
    self.last_action =action
    self.last_features = features

    return action

  def receive_game_start_message(self, game_info):
    pass

  def receive_round_start_message(self, round_count, hole_card, seats):
    pass

  def receive_street_start_message(self, street, round_state):
    pass

  def receive_game_update_message(self, new_action, round_state):
    pass

  def receive_round_result_message(self, winners, hand_info, round_state):
    if not self.last_action == None:
      self.run_updates(None,None,round_state,None)
    self.last_state = None
    self.last_reward = None
    self.last_features = None
    self.last_chose_greedy = None

  def run_updates(self,valid_actions,hole_card,round_state,features):
    has_last_state = (self.last_state != None)

    if has_last_state:

    #compute reward since last state
      reward= self.compute_reward(round_state)

      if not valid_actions == None and hole_card == None: #there will be an action this round.
        event = Event(self.last_state, self.last_features,self.last_action, reward,[valid_actions,hole_card,round_state],features,self.player_index)

      else: #this is the end.
        event = Event(self.last_state, self.last_features,self.last_action, reward,None,None,self.player_index)

    #add last state + reward to RL buffer
      self.buffer.add_to_buffer(event)
      if self.last_chose_greedy:
        self.bufferSL.add_to_buffer(event)
        

    #update RL network
    for ev in self.buffer.sample(3):
        self.network.update(ev,0.9,self.network)

    #update SL network    
    for ev in self.bufferSL.sample(5):
      self.self_network.update(ev,self.player_index)
      
    #periodically update target network
    self.iterations += 1
    if self.update_num != None:
      if self.iterations % self.update_num == 0:
        self.network.update_target_network()
