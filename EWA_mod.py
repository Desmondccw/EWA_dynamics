from matplotlib.pyplot import figure
import matplotlib.pyplot as plt

import numpy as np

from scipy import optimize
from scipy.optimize import minimize, Bounds
from scipy.integrate import solve_ivp
from scipy.optimize import approx_fprime
from scipy.linalg import expm

from scipy import linalg


''''1. Game Initialisation'''

''' Payoff Matrix Generation and initial points'''
'''Generate random matrices for the games'''

def payoff_tensor_generator_2_player(n_actions, gamma):
    '''Creates a payoff tensor with gamma correlations (for 2 players only)'''
    mean= np.zeros(2)
    cov= np.full((2,2), gamma )

    np.fill_diagonal(cov, 1 )

    a, b= np.random.multivariate_normal(mean, cov, (n_actions, n_actions)).T

    return np.array([a,b.T])

def payoff_tensor_generator(n_players, n_actions):
    '''n player variant... correlations not coupled will need to change it'''
    listy=[n_players]
    
    for i in range(n_players):
        listy.append(n_actions)
    return np.random.rand(*listy)

def generate_starting_point(n_players, n_actions):
    '''sets uniform distribution as starting strategy'''
    
    return np.ones((n_players, n_actions))/n_actions

def random_starting_point(n_players, n_actions):
    '''Random starting point'''

    nums= np.random.uniform(0,1,(n_players, n_actions))
    return nums / np.sum(nums, axis=1, keepdims=True)

''' Player class'''

class player:
    '''Generate player'''
    
    def __init__(self, index, state, self_tensor):
        
        self.index= index
        self.state= state
        self.tensor= self_tensor
        
    def calc_vector(self, full_state, r):

        '''Calculates update for a given player given memory term(r) and distributions of the the other players and itself (full state)'''
        
        round_update= self.tensor.copy()
        
        for count, i in enumerate(full_state):
            if count != self.index: 
                round_update= round_update @ np.array(i)
                
        decay= -np.log(np.abs(self.state) + 1e-28)/r        
        
        X= (round_update + decay)* self.state
        
        return X- np.sum(X)*self.state
    
    def mod_deriv(self, full_state, r):

        round_update= self.tensor.copy()
        
        for count, i in enumerate(full_state):
            if count != self.index: 
                round_update= round_update @ np.array(i)
                
        decay= -np.log(np.abs(self.state) + 1e-28)/r

        X= (round_update + decay)* self.state

        return round_update + decay - np.sum(X)
    
 ''' EWA_system class'''       
class EWA_system:

    '''A EWA system consisting of n_player, n_action depending (the dimensions are found via the tensor)'''
    
    def __init__(self, payoff_tensor, r):
        '''Parameters'''
        self.payoff_tensor= payoff_tensor
        self.time= 0
        
        self.r= r
        
        self.n_player, self.n_actions= payoff_tensor.shape[0], payoff_tensor.shape[-1]
        
        pass
    
    def initialise_state(self, state):
        
        self.state= state
    
    def update_vector(self):
        '''Calculates the derivative given a position'''
        
        update_vectors=[]
        
        for i in range(self.n_player):
            i_player= player(index= i, state= self.state[i], self_tensor= self.payoff_tensor[i])
            vect= i_player.calc_vector(full_state= self.state, r= self.r)
            update_vectors.append(vect)
            
        return np.array(update_vectors)
    
    def mod_derivative(self):
        
        update_vectors=[]
        
        for i in range(self.n_player):
            i_player= player(index= i, state= self.state[i], self_tensor= self.payoff_tensor[i])
            vect= i_player.mod_deriv(full_state= self.state, r= self.r)
            update_vectors.append(vect)
            
        return np.array(update_vectors)

    
    def simulate(self, steps, stepsize= 0.01):
        '''Constant step size integration, although flow map is preferred as it uses specialised ODE solvers (Runga-Kutta methods) '''
        states=[self.state]
        
        for s in range(steps):
            self.state= states[-1] + self.update_vector()* stepsize
            states.append(self.state)
            self.time+= stepsize
            
        return states

'''Derivative functions
    
    d_core computes the standard derivative
    d_core_mod computes derivative of x_i / x_i
    
    Useful to have functions for both'''
    
def d_core(t,X, payoff_tensor, r):
    '''Calculate the derivative given a state'''
    
    sys= EWA_system(payoff_tensor, r)
    
    X= np.array(X).reshape((sys.n_player, sys.n_actions))
    
    sys.initialise_state(X)
    
    vect= sys.update_vector().flatten()

    del sys
    
    for count, x in enumerate(X.flatten()):
        if x >= 1 and vect[count] >0:
            vect[count]=0
    return vect

def d_core_mod(t, X, payoff_tensor, r): 

    sys= EWA_system(payoff_tensor, r)
    
    X= np.array(X).reshape((sys.n_player, sys.n_actions))
    
    sys.initialise_state(X)
    
    vect= sys.mod_derivative().flatten()

    del sys
    
    for count, x in enumerate(X.flatten()):
        if x >= 1 and vect[count] >0:
            vect[count]=0
    return vect


def jacobian(t, X, payoff_tensor, r):
    '''Calculate the Jacobian given a state'''
    def f(x):
        return d_core(t, x, payoff_tensor, r)
    
    Jacobian = approx_fprime(X, f, epsilon=1e-6)
    return Jacobian

def calculate_exponents(tf, X0, payoff_tensor, r, maxstep=0.1):
    '''Calculates Lyapunov Exponents via QR decomposition'''

    '''Calculate the desired trajectory'''
    sol= solve_ivp(lambda t, y: d_core(t, y, payoff_tensor, r), t_span=(0, tf), y0=X0, max_step= maxstep)
    states, times = sol.y.T , sol.t

    y_0= np.eye(len(X0))
    time_diff= np.diff(times)

    Y=y_0
    exponents= np.zeros(len(X0))

    '''For each step in the trajectory:'''
    '''i) Calculate the Jacobian'''
    '''ii) QR decompose it'''
    '''iii) Calculate Q.T @ Jacobian @ Q (QJQ)'''
    ''''iv) Average out QJQ to find exponents'''

    for count, t_d in enumerate(time_diff):
        jacob= jacobian(0, states[count].flatten(), payoff_tensor, r)
        
        q, rr = linalg.qr(Y)
        
        exponents+= np.diag(q.T @ jacob @ q)*t_d

        Y= np.dot(expm(jacob * t_d), Y)

    return np.sort(exponents/tf)

def kaplan_yorke_dimension(exponents):
    ''' Calculates Kaplan Yorke Dimension given exponents'''
    '''Uses sort and cumsum'''
    sort= np.sort(exponents)[::-1]
    
    cumsum= np.cumsum(sort)
    cumsum_positive= [c for c in cumsum if c > 0]

    if len(cumsum_positive) < 1: 
        return 0
    
    else:
        return len(cumsum_positive)+ cumsum_positive[-1] / np.abs(sort[len(cumsum_positive)])

def flow_map(tf, X0, payoff_tensor, r, maxstep= 1):
    '''Integrates the system forwards in time given initial state X0 using Runga-Kutta 45'''

    '''Arbitrary max step size set to depend on the no. actions... more actions require smaller steps size as there is a higher chance of going out of bounds of (0,1)'''
    step_max= maxstep/ np.sqrt(payoff_tensor.shape[-1])
    
    sol= solve_ivp(lambda t, y: d_core(t, y, payoff_tensor, r), t_span=(0, tf), y0=X0, max_step= step_max)
    
    solve= sol.y.T[-1]
    return sol

def simulate_random_game(n_players= 2, n_actions= 20, gamma= -0.8, r= 500, time_interval=10000, visualise= True, est_KY_dim= True):
    a= payoff_tensor_generator_2_player(n_actions, gamma)
    initial_state= generate_starting_point(n_players, n_actions)

    sol= flow_map(time_interval, initial_state.flatten(), a, r)

    time_array, strategy_evolution_array= sol.t, sol.y

    output={ "time_array": time_array, 'strategy_evolution': strategy_evolution_array}

    if visualise== True:
        '''Visualise strategies of the first player'''

        fig1, ax1 = plt.subplots()
        plt.rcParams.update({'font.size': 12})

        ax1.stackplot(sol.t,sol.y[0:n_actions]);
        ax1.set_xlabel("time")
        ax1.set_ylabel('Probability for a given strategy')
        ax1.set_title('Probability evolution of strategies over time for player 1')
        ax1.set_xlim([0, time_interval])
        ax1.set_ylim([0, 1])
        
        fig1.set_size_inches(15,5)
        
        '''Visualise strategies of the second player'''
        fig2, ax2 = plt.subplots()
        
        plt.rcParams.update({'font.size': 12})
        ax2.stackplot(sol.t,sol.y[n_actions:]);
        ax2.set_xlabel("time")
        ax2.set_ylabel('Probability for a given strategy')
        ax2.set_title('Probability evolution of strategies over time for player 2')
        ax2.set_xlim([0, time_interval])
        ax2.set_ylim([0, 1])  
        fig2.set_size_inches(15,5)

    if est_KY_dim==True:
        '''Find KY dimension if desired'''
        new_time_interval= time_interval //3
        new_initial_state= sol.y.T[-new_time_interval]
        exp= calculate_exponents(new_time_interval, new_initial_state, a, r)

        ky= kaplan_yorke_dimension(exp)
        output.update({"Exponents": exp})
        output.update({"KY dimension": ky})
    return output

def find_fixed_point(X0, payoff_tensor, r):
    '''Finds a fixed point for a given game'''
    '''Dangerous does not gurantee stability as there might be multiple fixed_point'''

    '''Constraint to the simplex'''

    n_actions, n_players= payoff_tensor.shape[-1] , 2

    '''2 constraints...
    a) non-negativity
    b) Simplex constraint'''
    
    constraints =[{'type': 'eq', 'fun': lambda x: np.sum(x[:n_actions])-1}] + \
                [{'type': 'eq', 'fun': lambda x: np.sum(x[n_actions:])-1}]

    '''returns derivative'''
    def fun(x):
        return np.sum(d_core_mod(0, x, payoff_tensor, r)**2)

    sol = minimize(fun, X0, constraints=constraints, bounds= Bounds([0 for i in range(len(X0))], [1 for i in range(len(X0))]) )
    
    return sol

def run_till_convergence(n_players= 2, n_actions= 20, gamma= -0.8, r= 500, time_interval=100):

    tol = 1e-8
    count= 0

    converged = False

    a= payoff_tensor_generator_2_player(n_actions, gamma)
    initial_state= generate_starting_point(n_players, n_actions)

    def has_converged(state, tol):
        
        return np.linalg.norm(d_core(0, state, a, r)) < tol 
    
    state= initial_state.copy()

    while not converged and count < 60:
        # Solve the IVP for the current time span
        sol= flow_map(time_interval, state.flatten(), a, r)

        state= (sol.y.T[-1]).flatten()

        converged = has_converged(state, tol)

        count+=1

        if count > 10:
            print(f'Count: {count}')

    if converged:
        return state
    
    else:
        return np.array([np.inf])