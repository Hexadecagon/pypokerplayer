from Optimal_Raise_Qvalues import optimal_raise_Qvalues
from Processing import get_bot_info
from Raise_Sizes import find_index


import tensorflow as tf
import copy
import random
from numpy import argmax


class ReplayBuffer(object): #need different replaybuffers for different streets
    def __init__(self,size):
        self.size = size
        self.buffer = []
    def add_to_buffer(self,event):
        if len(self.buffer) >= self.size:
            self.buffer.remove(self.buffer[0])
        self.buffer.append(event)
    def sample(self,n):
        if n >= len(self.buffer):
            return self.buffer
        else:
            numbers_to_grab = [random.randrange(0,n) for z in range(n)]
            return [self.buffer[number] for number in numbers_to_grab]
        
class Event(object):
    def __init__(self,state0, features0, action, reward,state1,features1, player_index):
        self.state0 = copy.deepcopy(state0) #states in [valid_actions, hole_card,round_state]
        self.features0 = features0
        self.action = copy.deepcopy(action) #actions in [action, amount]
        self.reward = copy.deepcopy(reward)
        self.state1 = copy.deepcopy(state1)
        self.features1 = features1
        self.player_index = player_index
        

    def __str__(self):
        return str(self.state0)+str(self.action) + str(self.reward) + str(self.state1)
class DQN(object): #Implements Double DQN, with the slow-moving copy of the network instead of two separately trained ones. This is fine.
    def __init__(self, dim, u, name):

        self.sess = tf.Session()
        self.name = name

        self.x = tf.placeholder(tf.float32, [None,dim])
        self.L = []
        self.T = []
        for index,unit in enumerate(u):
            if index == 0:
                self.L.append(tf.layers.dense(inputs = self.x, units = unit, activation = tf.nn.relu, name = "L"+str(index)+str(name)))
                self.T.append(tf.layers.dense(inputs = self.x, units = unit, activation = tf.nn.relu, name = "T"+str(index)+str(name)))
            else:
                self.L.append(tf.layers.dense(inputs = self.L[index-1], units = unit, activation = tf.nn.relu, name = "L"+str(index)+str(name)))
                self.T.append(tf.layers.dense(inputs = self.T[index-1], units = unit, activation = tf.nn.relu, name = "T"+str(index)+str(name)))
        
        self.Qout = tf.layers.dense(inputs = self.L[len(self.L)-1], units = 27,name = "Qout"+str(name)) #fold, call/check, raise to n

        self.TQout = tf.layers.dense(inputs = self.L[len(self.L)-1], units = 27,name = "TQout"+str(name))

        self.saver = tf.train.Saver()
        self.target_Q = tf.placeholder(tf.float32)
        self.loss = (self.target_Q - self.Qout)**2


                                    
        self.opt = tf.train.RMSPropOptimizer(0.00001).minimize(self.loss)
        
        self.sess.run(tf.global_variables_initializer())
        
        self.update_target_network()
  #      tf.get_default_graph().finalize()


    def get_qs(self,features,target_mode):
        if target_mode == "original":
            Qnet = self.Qout
        elif target_mode == "target":
            Qnet = self.TQout
        return self.sess.run(Qnet,feed_dict = {self.x:[features]})
    
    def get_weights_and_biases(self,name):
        with tf.variable_scope(name,reuse = True):
            return self.sess.run(tf.get_variable("kernel")), self.sess.run(tf.get_variable("bias"))

    def update(self,event,gamma,after_network): #Uses target Q network.

        state_before = event.state0 #states
        state_after = event.state1
        player_index = event.player_index
        action = event.action

        features = event.features0

        normal_Qs = self.sess.run(self.Qout, feed_dict={self.x:[features]}).tolist()[0]#Calculate Qvalues of state 0
                                  
        if(state_after != None):
            optimal_Qs = optimal_raise_Qvalues(state_after[0],state_after[1],state_after[2],after_network,player_index,"target",features)[0] #calculate Qvalues of state 1
        else:
            optimal_Qs = [0] #If there is no state 1, like if it is the end of the hand, set the maximal next Qvalue to 0.
            #This may have to be reverted, but this DOES make more sense than setting the next Qvalue to the next hand. Which doesn't make sense.
            
        bellman = event.reward+gamma*max(optimal_Qs) #Bellman update.


        if(action[0] == "fold"): #apply Bellman update to appropriate qvalue of qs before change.
            normal_Qs[0] = bellman
        elif(action[0] == "call"):
            normal_Qs[1] = bellman
        elif (action[0] == "raise"):
            all_in = state_before[0][2]["amount"]["max"]
            min_bet = state_before[0][2]["amount"]["min"]
            index = find_index(min_bet,all_in,action[1])+2

            normal_Qs[index] = bellman
        self.sess.run(self.opt,feed_dict = {self.x:[features],self.target_Q:[normal_Qs]}) #run optimizer to set the correct q of features before, to the adjusted Qvalues.
                
    def upload_original_network(self,weights):
        names =  ["L" + str(n) + str(self.name) for n in range(len(self.L))] +["Qout" + str(self.name)]
        for index,scope in enumerate(names):
            with tf.variable_scope(scope,reuse=True):
                k = tf.get_variable("kernel")
                b = tf.get_variable("bias")
                assignment_k = k.assign(weights[index][0])
                assignment_b = b.assign(weights[index][1])
                self.sess.run(assignment_k)
                self.sess.run(assignment_b)

    def update_target_network(self): #IT WORKS, I THINK.

        print("dqn: updating target network")
        Ls = []


        for index in range(len(self.L)):
            Ls.append("L"+str(index)+str(self.name))

        old_weights  = [self.get_weights_and_biases(L) for L in Ls+ ["Qout"+str(self.name)]]
        
        names = ["T" + str(n) + str(self.name) for n in range(len(self.T))]  +["TQout"+str(self.name)]

        for index,scope in enumerate(names):
            with tf.variable_scope(scope,reuse=True):
                k = tf.get_variable("kernel")
                b = tf.get_variable("bias")

                k.load(old_weights[index][0],self.sess)
                b.load(old_weights[index][1],self.sess)
                

    def restore(self):
        print("dqn: restoring")
    #    self.saver = tf.train.import_meta_graph("./models/main/"+self.name+".meta")
        self.saver.restore(self.sess,tf.train.latest_checkpoint('./models/main/'))

        self.update_target_network()
    def save(self):
        print("dqn: saving")
        self.saver.save(self.sess,"models/main/" + self.name,write_meta_graph=False)
##class DQN(object): #Implements Double DQN, with the slow-moving copy of the network instead of two separately trained ones. This is fine.
##    def __init__(self, dim, u, name):
##
##        self.sess = tf.Session()
##        self.name = name
##
##        self.x = tf.placeholder(tf.float32, [None,dim])
##        self.LV = []
##        self.LA = []
##        
##        self.TV = []
##        self.TA = []
##        for index,unit in enumerate(u):
##            if index == 0:
##                self.LV.append(tf.layers.dense(inputs = self.x, units = unit, activation = tf.nn.relu, name = "LV"+str(index)+str(name)))
##                self.LA.append(tf.layers.dense(inputs = self.x, units = unit, activation = tf.nn.relu, name = "LA"+str(index)+str(name)))
##                self.TV.append(tf.layers.dense(inputs = self.x, units = unit, activation = tf.nn.relu, name = "TV"+str(index)+str(name)))
##                self.TA.append(tf.layers.dense(inputs = self.x, units = unit, activation = tf.nn.relu, name = "TA"+str(index)+str(name)))       
##            else:
##                self.LV.append(tf.layers.dense(inputs = self.LV[index-1], units = unit, activation = tf.nn.relu, name = "LV"+str(index)+str(name)))
##                self.TV.append(tf.layers.dense(inputs = self.TV[index-1], units = unit, activation = tf.nn.relu, name = "TV"+str(index)+str(name)))
##                self.LA.append(tf.layers.dense(inputs = self.LA[index-1], units = unit, activation = tf.nn.relu, name = "LA"+str(index)+str(name)))
##                self.TA.append(tf.layers.dense(inputs = self.TA[index-1], units = unit, activation = tf.nn.relu, name = "TA"+str(index)+str(name)))
##
##
##        self.value_function = tf.layers.dense(inputs=self.LV[len(self.LV)-1],units =1,name = "Value"+str(name))
##        self.advantage_function = tf.layers.dense(inputs = self.LA[len(self.LA)-1],units=27,name= "Advantage"+str(name))
##
##        self.Tvalue_function = tf.layers.dense(inputs=self.TV[len(self.TV)-1],units =1,name = "TValue"+str(name))
##        self.Tadvantage_function = tf.layers.dense(inputs = self.TA[len(self.TA)-1],units=27,name= "TAdvantage"+str(name))        
##        
##        self.Qout = self.value_function + self.advantage_function - tf.reduce_mean(self.advantage_function) #fold, call/check, raise to n
##        self.TQout = self.Tvalue_function + self.Tadvantage_function 
##
##        self.saver = tf.train.Saver()
##        self.target_Q = tf.placeholder(tf.float32)
##        self.loss = (self.target_Q - self.Qout)**2
##
##        self.opt = tf.train.RMSPropOptimizer(0.00001).minimize(self.loss)
##        
##        self.sess.run(tf.global_variables_initializer())
##        
##        self.update_target_network()
##
##    def update(self,event,gamma,after_network): #Uses target Q network.
##
##        state_before = event.state0 #states
##        state_after = event.state1
##        player_index = event.player_index
##        action = event.action
##
##        features = event.features0
##
##        normal_Qs = self.sess.run(self.Qout, feed_dict={self.x:[features]}).tolist()[0]#Calculate Qvalues of state 0
##                                  
##        if(state_after != None):
##            optimal_Qs = optimal_raise_Qvalues(state_after[0],state_after[1],state_after[2],after_network,player_index,"target",features)[0] #calculate Qvalues of state 1
##        else:
##            optimal_Qs = [0] #If there is no state 1, like if it is the end of the hand, set the maximal next Qvalue to 0.
##            #This may have to be reverted, but this DOES make more sense than setting the next Qvalue to the next hand. Which doesn't make sense.
##            
##        bellman = event.reward+gamma*max(optimal_Qs) #Bellman update.
##
##
##        if(action[0] == "fold"): #apply Bellman update to appropriate qvalue of qs before change.
##            normal_Qs[0] = bellman
##        elif(action[0] == "call"):
##            normal_Qs[1] = bellman
##        elif (action[0] == "raise"):
##            all_in = state_before[0][2]["amount"]["max"]
##            min_bet = state_before[0][2]["amount"]["min"]
##            index = find_index(min_bet,all_in,action[1])+2
##
##            normal_Qs[index] = bellman
##        self.sess.run(self.opt,feed_dict = {self.x:[features],self.target_Q:[normal_Qs]}) #run optimizer to set the correct q of features before, to the adjusted Qvalues.
##
##    def get_qs(self,features,target_mode):
##        if target_mode == "original":
##            Qnet = self.Qout
##        elif target_mode == "target":
##            Qnet = self.TQout
##        return self.sess.run(Qnet,feed_dict = {self.x:[features]})
##    
##    def get_weights_and_biases(self,name):
##        with tf.variable_scope(name,reuse = True):
##            return self.sess.run(tf.get_variable("kernel")), self.sess.run(tf.get_variable("bias"))
##
##    def update_target_network(self): #IT WORKS, I THINK.
##
##        print("dqn: updating target network")
##        LVs = []
##        LAs = []
##        for index in range(len(self.LV)):
##            LVs.append("LV"+str(index)+str(self.name))
##        for index in range(len(self.LA)):
##            LAs.append("LA"+str(index)+str(self.name))
##
##        old_weights  = [self.get_weights_and_biases(L) for L in LVs+ LAs + ["Value"+str(self.name),"Advantage"+str(self.name)]]
##        
##        names = ["TV" + str(n) + str(self.name) for n in range(len(self.TV))] + ["TA" + str(n) + str(self.name) for n in range(len(self.TA))] +["TValue"+str(self.name),"TAdvantage"+str(self.name)]
##
##        for index,scope in enumerate(names):
##            with tf.variable_scope(scope,reuse=True):
##                k = tf.get_variable("kernel")
##                b = tf.get_variable("bias")
##
##                k.load(old_weights[index][0],self.sess)
##                b.load(old_weights[index][1],self.sess)
##                
##    def restore(self):
##        print("dqn: restoring")
##    #    self.saver = tf.train.import_meta_graph("./models/main/"+self.name+".meta")
##        self.saver.restore(self.sess,tf.train.latest_checkpoint('./models/main/'))
##
##        self.update_target_network()
##    def save(self):
##        print("dqn: saving")
##        self.saver.save(self.sess,"models/main/" + self.name,write_meta_graph=False)
##        
