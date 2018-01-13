import numpy as np
import gridrender as gui
import matplotlib.pyplot as plt

class ValueLearning:
    def __init__(self, ind, GridWorld, policy, max_iter=600):
        self.id = ind
        self.max_iter = max_iter
        self.GridWorld = GridWorld
        self.policy = policy
        
        self.trajRewards = np.zeros((self.GridWorld.n_states,np.size(self.GridWorld.action_names),self.max_iter))
        self.SAV = np.zeros((self.GridWorld.n_states,np.size(self.GridWorld.action_names),self.max_iter))
        self.V = np.zeros((self.GridWorld.n_states,self.max_iter))
        self.J = np.zeros((self.max_iter))
        self.GridWorld.render = False
        
    def run(self):
        """Runs the estimation based on GridWorld, policy and max_iter"""
        for ind in range(self.max_iter):
            # For each new trajectory, we start by computing a new initial state
            x0 = self.GridWorld.reset() #Return state ID of random initial state
            for a0 in self.GridWorld.state_actions[x0]:
                cumRew = 0
                term = False
                
                [x,r,term] = self.GridWorld.step(x0,a0)
                cumRew = r
                
                step = 1
                while not term:
                    [x,r,term] = self.GridWorld.step(x,self.policy(x,self.GridWorld))
                    cumRew = cumRew + np.power(self.GridWorld.gamma,step)*r
                    step += 1
                self.trajRewards[x0,a0,ind] = cumRew
                
                
            for x in range(self.GridWorld.n_states):
                for a in range(len(self.GridWorld.action_names)):
                    trajCount = np.sum(self.trajRewards[x][a]!=0)

                    if self.trajRewards[x,a,ind] != 0:
                        self.SAV[x,a,ind] = ((trajCount-1)/trajCount)*self.SAV[x,a,ind-1] + (1/(trajCount))*self.trajRewards[x,a,ind]
                    elif ind>0 and trajCount>0:
                        self.SAV[x,a,ind] = self.SAV[x,a,ind-1]
                    else:
                        self.SAV[x,a,ind] = self.trajRewards[x,a,ind]
                        
                self.V[x,ind] = self.SAV[x,self.policy(x,self.GridWorld),ind]
                
            # print("Traj reward slice at time {0:2d}\n".format(ind))
            # print(self.trajRewards[:,:,ind])
            # print("\n SAV at time {0:2d}\n".format(ind))
            # print(self.SAV[:,:,ind])
            # print("\n Value at time {0:2d}\n".format(ind))
            # print(self.V[:,ind])
            
        self.J = np.average(self.V,axis=0)
    
    def plotJDiff(self, RefValue):
        """Plots the evolution of J-value with number of trajectories ran"""
        JRef = np.average(RefValue,axis=0)
        Diff = self.J - JRef
        f, axarr = plt.subplots(1)
        axarr.plot(Diff)
        axarr.set_title('Average value error to optimal')
        axarr.set_xlabel('Episode')
        axarr.set_ylabel('Average error (over all states)')
        
    def render(self):
        """Renders the Q-function learned"""
        gui.render_q(self.GridWorld, self.SAV[:,:,self.max_iter-1])     


class PolicyLearning:
    def __init__(self, ind, GridWorld, max_iter=5000):
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
        
    def run(self, optValue=None, record_trajectory = []):
        """Runs the estimation based on GridWorld, policy and max_iter"""
        """ Tristan adds: record_trajectory in order to compute the DDs for option discovery """
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
            
            record_trajectory.append([])
            
            while not term:      
                action = self.explPol(state,time)
                self.counter[state,action] += 1
                [newState,r,term] = self.GridWorld.step(state,action)
                
                record_trajectory[-1].append((state,action,r))
                
                rew += r
                tempDiff = r + self.GridWorld.gamma * max(self.SAV[newState,:,time]) - self.SAV[state,action,time]
                self.SAV[state,action,time] = self.SAV[state,action,time] + self.learningRate(state,action)*tempDiff
                state = newState
            
            self.value[:,time] = np.max(self.SAV[:,:,time],axis=1)
            self.cumReward[time] += rew
        
        if optValue is not None:
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