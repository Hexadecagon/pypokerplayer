
from DQObjects import DQN, Event,ReplayBuffer,SelfActionNN
import tensorflow as tf
import pypokerengine
import random

from DQPlayer import DQPlayer
from examples.players.console_player import ConsolePlayer
from examples.players.fish_player import FishPlayer
from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.api.emulator import Emulator



sess = tf.Session()


dqn_0 =DQN(124,[1024,1024,1024,1024,1024],"main")
self_action_NN = SelfActionNN(123,[512,512,512,512], "self_action")

bufferRL = ReplayBuffer(10000)
bufferSL = ReplayBuffer(10000)

sess.run(tf.global_variables_initializer())

t = 0


params = [0,0.8,0.000025,0,None] #epsilon, anticipatory, ,decay,min epsilon, update num
try:
  dqn_0.restore()
  self_action_NN.restore()
except FileNotFoundError:
  pass

player1 = DQPlayer(dqn_0,self_action_NN,bufferRL,bufferSL,0,params)
player2 = ConsolePlayer()

for q in range(1000000):
    try:
      config = setup_config(max_round= 100, initial_stack = 100, small_blind_amount = 5)
      config.register_player(name= "Player1", algorithm = player1)
      config.register_player(name="Me", algorithm = player2)
      game_result = start_poker(config,verbose=1)
    except ValueError:
      pass
    print("Epsilon " + str(player1.epsilon))
    print("Iteration " + str(player1.iterations))
##    if q % 100 == 0:
##      dqn_0.save()
##      self_action_NN.save()
