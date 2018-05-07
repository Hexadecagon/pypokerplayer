
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


dqn_0 =DQN(124,[1024,512,1024,512],"main")
self_action_NN = SelfActionNN(123,[1024,512,1024,512], "self_action")

bufferRL = ReplayBuffer(60000)
bufferSL = ReplayBuffer(300000)

sess.run(tf.global_variables_initializer())

t = 0

##try:
##  dqn_0.restore()
##  self_action_NN.restore()
##except FileNotFoundError:
##  pass
player1 = DQPlayer(dqn_0,self_action_NN,bufferRL,bufferSL,0)
for q in range(1000000):

    config = setup_config(max_round= 50, initial_stack = 100, small_blind_amount = 5)
    config.register_player(name= "Player1", algorithm = player1)
    config.register_player(name="Me", algorithm = DQPlayer(dqn_0,self_action_NN,bufferRL,bufferSL,1))
    game_result = start_poker(config,verbose=1)

    print("Epsilon " + str(player1.epsilon))
    print("Iteration " + str(player1.iterations))
    if q % 10 == 0:
      dqn_0.save()
      self_action_NN.save()
