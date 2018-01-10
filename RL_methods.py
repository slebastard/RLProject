class PolicyLearning:
    def __init__(self, ind, GridWorld, max_iter=1000):
        self.id = ind
        self.max_iter = max_iter
        self.GridWorld = GridWorld
        self.max_exploration = 0.4
        self.min_exploration = 0.1
        
        self.trajRewards = np.zeros((self.GridWorld.n_states,np.size(self.GridWorld.action_names),self.max_iter))
        # We must make sure the unreachable states always have a lower Q-value than the worst case scenario estimate
        # for that the exploration policy estimation to work
        self.SAV = np.zeros((self.GridWorld.n_states,np.size(self.GridWorld.action_names),self.max_iter))
        self.value = np.zeros((self.GridWorld.n_states,self.max_iter))
        self.maxValueError = np.zeros((self.max_iter))
        self.GridWorld.render = False

    def explorationRate(self, time):
        return self.min_exploration + max((self.max_iter-2*time)/self.max_iter,0)*(self.max_exploration - self.min_exploration)
        
    def explPol(self, state, time):
        v = -1000*np.ones(len(self.GridWorld.action_names))
        for i in self.GridWorld.state_actions[state]:
            v[i] = self.SAV[state,i,time]
        optAction = np.argmax(v)
        failed = np.random.rand(1) < self.explorationRate(time)
        if failed or np.isnan(optAction):
            return np.random.choice(self.GridWorld.state_actions[state])
        else:
            return optAction

    def learningRate(self, state, action):
        """For simplicity, I will first implement the harmonic learning rate (dynamic average)"""
        trajCount = self.counter[state, action]
        if trajCount == 0:
            return 1
        else:
            return 1/(trajCount)
        
    def run(self, optValue):
        """Runs the estimation based on GridWorld, policy and max_iter"""
        self.counter = np.zeros((self.GridWorld.n_states,len(self.GridWorld.action_names)))
        self.cumReward = np.zeros((self.max_iter))
        for time in range(self.max_iter):
            # For each new trajectory, we start by computing a new initial state
            initState = self.GridWorld.reset() #Return state ID of random initial state
            term = False
            state = initState
            rew = 0
            
            if time > 0:
                self.SAV[:,:,time] = self.SAV[:,:,time-1]
                self.cumReward[time] = self.cumReward[time-1]
            
            while not term:      
                action = self.explPol(state,time)
                self.counter[state,action] += 1
                [newState,r,term] = self.GridWorld.step(state,action)
                rew += r
                tempDiff = r + self.GridWorld.gamma * max(self.SAV[newState,:,time]) - self.SAV[state,action,time]
                self.SAV[state,action,time] = self.SAV[state,action,time] + self.learningRate(state,action)*tempDiff
                state = newState
             
            self.value[:,time] = np.max(self.SAV[:,:,time],axis=1)
            self.cumReward[time] += rew
        
        self.maxValueError = np.max(np.abs(self.value - optValue.reshape((self.GridWorld.n_states,1))), axis=0)
        self.meanCumReward = np.mean(self.cumReward)
        
        # Evaluating optimal policy
        self.policy = np.argmax(self.SAV[:,:,self.max_iter-1],axis=1)
        
    def stats(self):
        print("Stats from last run")
        f, axarr = plt.subplots(2, sharex=True)
        axarr[0].plot(self.maxValueError)
        axarr[0].set_title('Maximum error in between value estimation and optimal value')
        axarr[0].set_xlabel('Episode')
        axarr[0].set_ylabel('Max error (abs)')
        axarr[1].plot(self.cumReward)
        axarr[1].set_title('Cumulative reward over episodes')
        axarr[1].set_xlabel('Episode')
        axarr[1].set_ylabel('Cumulative reward')
        
        print("Mean reward cumulated over single episode: {0:.2f}".format(self.meanCumReward))
        
    def render(self):
        """Renders the Q-function learned"""
        gui.render_q(self.GridWorld, self.SAV[:,:,self.max_iter-1])  