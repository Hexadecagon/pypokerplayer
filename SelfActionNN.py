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
        
        self.opt = tf.train.RMSPropOptimizer(lrate).minimize(self.loss)
        self.opt_1 = tf.train.RMSPropOptimizer(lrate).minimize(self.loss_1)

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        
   #     tf.get_default_graph().finalize()

        
    def update(self,event,player_index): #Uses target Q network.
#        print("self action: nodes" + str(len([n.name for n in tf.get_default_graph().as_graph_def().node])))
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
    #        print(amount)
        return (action,amount)

    def get_weights_and_biases(self,name):
        with tf.variable_scope(name,reuse = True):
            return self.sess.run(tf.get_variable("kernel")), self.sess.run(tf.get_variable("bias"))

    def restore(self):
        print("self net: restoring")
     #   self.saver = tf.train.import_meta_graph("./models/self_action/"+self.name+".meta")
        self.saver.restore(self.sess,tf.train.latest_checkpoint('./models/self_action/'))
        
    def save(self):
        print("self net: saving")
        self.saver.save(self.sess,"models/self_action/" + self.name,write_meta_graph=False)
