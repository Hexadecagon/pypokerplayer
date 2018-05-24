
from DQObjects import DQN, Event,ReplayBuffer,SelfActionNN
import tensorflow as tf
import pypokerengine
import random

from DQPlayer import DQPlayer
from  examples.players.honest_player import HonestPlayer
from  examples.players.random_player import RandomPlayer
from pypokerengine.api.game import setup_config, start_poker


i = 20
dqn_0 =DQN(123,[1024,1024,1024,1024],"main")
self_action_NN = SelfActionNN(123,[512,512,512,512], "self_action")

bufferRL = ReplayBuffer(10000)
bufferSL = ReplayBuffer(1000)

params = [0.1,1,0,0,None] #epsilon, anticipatory, ,decay,min epsilon, update num
try:
  dqn_0.restore()
  self_action_NN.restore()
except FileNotFoundError:
  print("Error!")
  
player1 = DQPlayer(dqn_0,self_action_NN,bufferRL,bufferSL,0,params,True)
player2 =RandomPlayer()#HonestPlayer()

def run_simulation(i,player1,player2):

    wins = 0
    losses = 0
    for q in range(i):
      print(q)
      try:
        config = setup_config(max_round= 30, initial_stack = 100, small_blind_amount = 1)
        config.register_player(name= "Player1", algorithm = player1)
        config.register_player(name="Player2", algorithm = player2)
        game_result = start_poker(config,verbose=0)

        results = game_result["players"][0]["stack"] > game_result["players"][1]["stack"]
        if results == True:
              wins += 1
        else:
              losses += 1
      except ValueError:
        pass
      print(wins,losses)
    return wins/(wins+losses)

print(run_simulation(50,player1,player2))
