"""
Description:
    You are going to implement Dyna-Q, a integration of model-based and model-free methods. 
    Please follow the instructions to complete the assignment.
"""
import numpy as np 
from copy import deepcopy as cp
import itertools

def choose_action(state, q_value, maze, epislon):
    """
    Description:
        choose the action using epislon-greedy policy
    """
    if np.random.random() < epislon:
        return np.random.choice(maze.actions)
    else:
        values = q_value[state[0], state[1], :]
        return np.random.choice([action for action, value in enumerate(values) if value == np.max(values)])

def dyna_q(args, q_value, model, maze):
    """
    Description:
        Dyna-Q algorithm is here :)
    Inputs:
        args:    algorithm parameters
        q_value: Q table to maintain.
        model:   The internal model learned by Dyna-Q 
        maze:    Maze environment
    Return:
        steps:   Total steps taken in an episode.
    TODO:
        Complete the algorithm.
    """
    # (a) S ← current (nonterminal) state
    state = cp(maze.START_STATE)
        
    for steps in itertools.count():
        
        # (b)  A ← ε-greedy(S, Q)
        action=choose_action(state, q_value, maze, args.epislon)
        
        # (c) Execute action A; observe resultant reward, R, and state, S′ 
        next_state,R=maze.step(state,action) 
                
        # (d) Q(S, A) ← Q(S, A) + α [ R + γ max a Q(S′, a) − Q(S, A) ]
        if next_state in  maze.GOAL_STATES:
           q_value[state[0]][state[1]][action]+= args.alpha*(R- q_value[state[0]][state[1]][action])
           break
        
           
            
        q_value[state[0]][state[1]][action]+= args.alpha*(R + args.gamma*q_value[next_state[0]][next_state[1]][np.argmax(q_value[next_state[0]][next_state[1]])] - q_value[state[0]][state[1]][action])
        
        # (e)  Q(S, A) ← Q(S, A) + α [ R + γ max a Q(S′, a) − Q(S, A) ]
        model.store(state, action, next_state, R)
        
        # Update state
        state=cp(next_state)
     
        
        #update steps
        
        # (f) Repeat n times:
        for _ in range(args.planning_steps):
            
            # S ← random previously observed state # A ← random action previously taken in S   
            state_m, action_m, next_state_m, reward_m=model.sample()
            
            # R,S′←Model(S,A)
            q_value[state_m[0]][state_m[1]][action_m]+= args.alpha*(reward_m + args.gamma*q_value[next_state_m[0]][next_state_m[1]][np.argmax(q_value[next_state_m[0]][next_state_m[1]])] - q_value[state_m[0]][state_m[1]][action_m])
    
  
    return steps+1

class InternalModel(object):
    """
    Description:
        We'll create a tabular model for our simulated experience. Please complete the following code.
    """
    def __init__(self):
        self.model =dict()
        self.rand = np.random
        # Initialize the dictionary with state action next state and next action
        self.model={'State':[],'Action':[],'Next_state':[], 'reward':[]}
        
    def store(self, state, action, next_state, reward):
        """
        TODO:
            Store the previous experience into the model.
        Return:
            NULL
        """
        
        self.model['State'].append(state)
        self.model['Action'].append(action)
        self.model['Next_state'].append(next_state)
        self.model['reward'].append(reward)
        

    def sample(self):
        """
        TODO:
            Randomly sample previous experience from internal model.
        Return:
            
        """
        y=self.rand.choice(len(self.model['State']))
        
        x=self.rand.choice( [a for a in range(len(self.model['State']))  if self.model['State'][a] == self.model['State'][y] ] )
        
        return  self.model['State'][y],self.model['Action'][x],self.model['Next_state'][x],self.model['reward'][x]      
