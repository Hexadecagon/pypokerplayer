from Optimal_Raise_Qvalues import optimal_raise_Qvalues
from Processing import get_bot_info
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
            b = copy.deepcopy(self.buffer)
            random.shuffle(b)
            return self.buffer[:n]
        
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

        self.Qout = tf.layers.dense(inputs = self.L[len(self.L)-1], units = 3,name = "Qout"+str(name)) #fold, call/check, raise to n

        self.TQout = tf.layers.dense(inputs = self.L[len(self.L)-1], units = 3,name = "TQout"+str(name))

        self.saver = tf.train.Saver()
        self.target_Q = tf.placeholder(tf.float32)
        self.loss = (self.target_Q - self.Qout)**2
            
        self.opt = tf.train.AdamOptimizer(0.00001).minimize(self.loss)
        
        self.sess.run(tf.global_variables_initializer())
        
        self.update_target_network()


    def update(self,event,gamma,after_network): #Uses target Q network.
        
        state_before = event.state0 #states
        state_after = event.state1
        player_index = event.player_index
        action = event.action
        
      #  features = get_bot_info(state_before[0],state_before[1],state_before[2],player_index)+[action[1]] #state0: state+action
        features = event.features0+ [action[1]] 

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
            normal_Qs[2] = bellman

        self.sess.run(self.opt,feed_dict = {self.x:[features],self.target_Q:[normal_Qs]}) #run optimizer to set the correct q of features before, to the adjusted Qvalues.

    def get_qs(self,features,target_mode):
        if target_mode == "original":
            Qnet = self.Qout
        elif target_mode == "target":
            Qnet = self.TQout
        return self.sess.run(Qnet,feed_dict = {self.x:[features]})
    
    def get_weights_and_biases(self,name):
        with tf.variable_scope(name,reuse = True):
            return self.sess.run(tf.get_variable("kernel")), self.sess.run(tf.get_variable("bias"))

    def update_target_network(self): #IT WORKS, I THINK.
        Ls = []
        for index in range(len(self.L)):
            Ls.append(self.get_weights_and_biases("L"+str(index)+str(self.name)))

        Qout = self.get_weights_and_biases("Qout"  + str(self.name))

        old_weights = Ls+[Qout]
        
        names = ["T" + str(n) + str(self.name) for n in range(len(self.L))] +["TQout" + str(self.name)]
    
        for index,scope in enumerate(names):
            with tf.variable_scope(scope,reuse=True):
                k = tf.get_variable("kernel")
                b = tf.get_variable("bias")
                
                assignment_k = k.assign(old_weights[index][0])
                assignment_b = b.assign(old_weights[index][1])
                self.sess.run(assignment_k)
                self.sess.run(assignment_b)
                
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

    def restore(self):
        names = ["L" + str(n) + str(self.name) for n in range(len(self.L))] +["Qout" + str(self.name)]
        all_weights = []
        for index,variable in enumerate(names):
            w = open("models/"+variable+"_weights.txt","r")
            b = open("models/" +variable + "_biases.txt","r")
            weights = eval(w.read())
            biases = eval(b.read())
            all_weights.append([weights,biases])
            w.close()
            b.close()
        self.upload_original_network(all_weights)
        self.update_target_network()
        
    def save(self):
        names = ["L" + str(n) + str(self.name) for n in range(len(self.L))] +["Qout" + str(self.name)]
        for index,variable in enumerate(names):
            w = open("models/"+variable+"_weights.txt","w")
            b = open("models/" +variable + "_biases.txt","w")
            weights,biases = self.get_weights_and_biases(variable)
            w.write(str(weights.tolist()))
            b.write(str(biases.tolist()))
            w.close()
            b.close()


class SelfActionNN( object):   
    def __init__(self, dim, u, name):
        self.sess = tf.Session()
        self.name = name

        lrate = 0.00001

        self.x = tf.placeholder(tf.float32, [None,dim])
        
        self.L = []
        self.amount_L = []
        
        for index,unit in enumerate(u):
            if index == 0:
                self.L.append(tf.layers.dense(inputs = self.x, units = unit, activation = tf.nn.relu, name = "L"+str(index)+str(name)))
                self.amount_L.append(tf.layers.dense(inputs = self.x, units = unit, activation = tf.nn.relu, name = "amount_L"+str(index)+str(name)))
            else:
                self.L.append(tf.layers.dense(inputs = self.L[index-1], units = unit, activation = tf.nn.relu, name = "L"+str(index)+str(name)))
                self.amount_L.append(tf.layers.dense(inputs = self.L[index-1], units = unit, activation = tf.nn.relu, name = "amount_L"+str(index)+str(name)))

        self.output = tf.layers.dense(inputs = self.L[len(self.L)-1], units = 3,name = "ActionProbs"+str(name),activation = tf.nn.softmax) #fold, call/check, raise, to n
        self.amount= tf.layers.dense(inputs = self.amount_L[len(self.amount_L)-1], units = 1,name = "Amount"+str(name))
        
        self.target_action = tf.placeholder(tf.float32)
        self.target_amount = tf.placeholder(tf.float32)
        
        self.loss = tf.losses.softmax_cross_entropy(self.target_action,self.output)
        self.loss_1 = tf.square(self.target_amount-self.amount)
        
        self.opt = tf.train.AdamOptimizer(lrate).minimize(self.loss)
        self.opt_1 = tf.train.AdamOptimizer(lrate).minimize(self.loss_1)

        
        self.sess.run(tf.global_variables_initializer())
        
    def update(self,event,player_index): #Uses target Q network.
        state_before = event.state0 #states
        state_after = event.state1
        action = event.action
        
    #    features = get_bot_info(state_before[0],state_before[1],state_before[2],player_index)
        features = event.features0

        if action[0] == "fold":
            correct = [1,0,0]
        elif action[0] == "call":
            correct = [0,1,0]
        elif action[0] == "raise":
            correct = [0,0,1]
        
        self.sess.run(self.opt,feed_dict={self.x:[features],self.target_action:[correct]})
        self.sess.run(self.opt_1, feed_dict = {self.x:[features],self.target_amount:[action[1]]})
        
    def predict(self,state0,player_index,feat = None):
        if feat == None:
            features = get_bot_info(state0[0],state0[1],state0[2],player_index)
        else:
            features = feat
        action = self.sess.run(self.output,feed_dict = {self.x:[features]}).tolist()[0]
        action_index = argmax(action)
        if action_index == 0:
            action = "fold"
            amount = 0.0
        elif action_index == 1:
            action = "call"
            amount = state0[0][1]["amount"]
        elif action_index == 2:
            action = "raise"
            amount = self.sess.run(self.amount,feed_dict = {self.x:[features]}).tolist()[0][0]
            print(amount)
        return (action,amount)

    def get_weights_and_biases(self,name):
        with tf.variable_scope(name,reuse = True):
            return self.sess.run(tf.get_variable("kernel")), self.sess.run(tf.get_variable("bias"))

    def upload_original_network(self,weights):
        names =  ["L" + str(n) + str(self.name) for n in range(len(self.L))] +["ActionProbs" + str(self.name)]
        for index,scope in enumerate(names):
            with tf.variable_scope(scope,reuse=True):
                k = tf.get_variable("kernel")
                b = tf.get_variable("bias")
                assignment_k = k.assign(weights[index][0])
                assignment_b = b.assign(weights[index][1])
                self.sess.run(assignment_k)
                self.sess.run(assignment_b)

    def restore(self):
        names = ["L" + str(n) + str(self.name) for n in range(len(self.L))] +["ActionProbs" + str(self.name)]
        all_weights = []
        for index,variable in enumerate(names):
            w = open("models/"+variable+"_weights.txt","r")
            b = open("models/" +variable + "_biases.txt","r")
            weights = eval(w.read())
            biases = eval(b.read())
            all_weights.append([weights,biases])
            w.close()
            b.close()
        self.upload_original_network(all_weights)
        
    def save(self):
        names = ["L" + str(n) + str(self.name) for n in range(len(self.L))] +["ActionProbs" + str(self.name)]
        for index,variable in enumerate(names):
            w = open("models/"+variable+"_weights.txt","w")
            b = open("models/" +variable + "_biases.txt","w")
            weights,biases = self.get_weights_and_biases(variable)
            w.write(str(weights.tolist()))
            b.write(str(biases.tolist()))
            w.close()
            b.close()
