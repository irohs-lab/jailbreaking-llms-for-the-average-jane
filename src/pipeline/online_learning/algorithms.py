import config.main
from src.utils.regression_oracles import BaseRegressionOracle
import pandas as pd
import numpy as np
import scipy as sp
import os
from tqdm import tqdm

def normalize_weights(w:np.ndarray)->np.ndarray:
    w_sum = np.sum(w)
    if w_sum == 0:
        return np.ones_like(w) / len(w)
    return w / w_sum

def compute_kshot_success_from_rewards(reward_row, actions):
    """
    reward_row : (n,) array of rewards (0 or 1)
    actions    : list/array of chosen actions

    Success if ANY action succeeds.
    """
    return 1 if reward_row[actions].max() > 0 else 0

def learn_weights(alg:str):
    return globals()[f"learn_weights_{alg}"]

def learn_weights_randomized_weighted_majority(cfg:config.main.ProjectConfig, data:dict[str,pd.DataFrame], expert_data:dict[str,np.ndarray], T:int, T_prime:int, n:int, _w:np.ndarray, lr:np.float64, eval_split:str):

    """

        Littlestone & Warmuth, 1994: The Weighted Majority Algorithm

    """

    expected_loss = {
        'train': np.zeros(T),
        eval_split:np.zeros(T_prime-T)
    }

    weight_log = {
        'train': np.zeros((T,n)),
        eval_split: np.zeros((T_prime-T, n))
    }

    observed_asr = {
        'train':0,
        eval_split:0
    }
    rng = np.random.default_rng(seed=cfg.seed)
    tr_exp_data = expert_data['train']
    te_exp_data = expert_data[eval_split]

    num_passes = cfg.attack.num_passes

    # Train Loop
    for t in range(T):
        loss_t = tr_exp_data[t,:]
        w = normalize_weights(_w)

        weight_log['train'][t,:] = w

        i_t = rng.choice(n,p=w)
        r_it = 1-loss_t[i_t]
        expected_loss['train'][t] = w.T@loss_t

        _w = _w*np.exp(-lr*loss_t)

        observed_asr['train']+=r_it
    
    rng = np.random.default_rng(seed=cfg.seed)
    # Test Loop
    for t in range(T_prime-T):
        loss_t = te_exp_data[t,:]
        w = normalize_weights(_w)

        weight_log[eval_split][t,:] = w

        if num_passes == 1:
            i_t = rng.choice(n,p=w)

            r_it = 1-loss_t[i_t]
        else:
            actions = rng.choice(n, size=num_passes, replace=False, p=w)
            r_it = compute_kshot_success_from_rewards(1-loss_t, actions)
        
        expected_loss[eval_split][t] = w.T@loss_t

        if cfg.attack.continual:
            _w = _w*np.exp(-lr*loss_t)
        
        observed_asr[eval_split]+=r_it

    if T>0:
        observed_asr['train']/=T
    observed_asr[eval_split]/=(T_prime-T)

    return dict(data=data,expert_data=expert_data,observed_asr=observed_asr, weight_log=weight_log)


def learn_weights_exp3(cfg:config.main.ProjectConfig, data:dict[str,pd.DataFrame], expert_data:dict[str,np.ndarray], T:int, T_prime:int,n:int,_w:np.ndarray, lr:np.float64, eval_split:str):

    """

        Auer, Cesa-Bianchi, Freund and Schapire, 2002: The non-Stochastic Multiarmed Bandit Problem
    
    """

    observed_asr = {
        'train':0,
        eval_split:0
    }

    weight_log = {
        'train':np.zeros((T,n)),
        eval_split:np.zeros((T_prime-T,n))
    }
    
    rng = np.random.default_rng(seed=cfg.seed)
    tr_exp_data = expert_data['train']
    te_exp_data = expert_data[eval_split]

    num_passes = cfg.attack.num_passes

    # Train loop
    for t in range(T):
        reward_t = 1-tr_exp_data[t,:]
        z_t = np.sum(_w)
        p_t = ((1-lr)/z_t)*_w + lr/n

        weight_log['train'][t,:] = p_t
        
        i_t = rng.choice(n, p=p_t)

        r_it = reward_t[i_t]
        r_it_hat = r_it/p_t[i_t]

        _w[i_t] = _w[i_t]*np.exp(lr*r_it_hat*(1/n))
        
        observed_asr['train']+=r_it
    
    rng = np.random.default_rng(seed=cfg.seed)

    #Test Loop
    for t in range(T_prime-T):
        reward_t = 1-te_exp_data[t,:]
        z_t = np.sum(_w)
        p_t = ((1-lr)/z_t)*_w + lr/n

        weight_log[eval_split][t,:] = p_t

        if num_passes == 1:
            i_t = rng.choice(n, p=p_t)

            r_it_learn = reward_t[i_t]
        else:
            actions = rng.choice(n,p=p_t, size=num_passes, replace=False)
            i_t = actions[0]
            r_it_learn = reward_t[i_t]
            r_it = compute_kshot_success_from_rewards(reward_t,actions)
        
        r_it_hat = r_it_learn/p_t[i_t]

        if cfg.attack.continual:
            _w[i_t] = _w[i_t]*np.exp(lr*r_it_hat*(1/n))
        
        if num_passes == 1:
            observed_asr[eval_split]+=r_it_learn
        else:
            observed_asr[eval_split]+=r_it

    if T>0:
        observed_asr['train']/=T
    observed_asr[eval_split]/=(T_prime-T)

    return dict(
        data=data,
        expert_data=expert_data,
        observed_asr=observed_asr,
        weight_log=weight_log
    )

def learn_weights_linear_cb(cfg:config.main.ProjectConfig, data:dict[str,pd.DataFrame], expert_data:dict[str,np.ndarray], T:int, T_prime:int,n:int,kappa:np.float64, alpha:np.float64, context_vectors:dict[str,np.ndarray], w:np.ndarray, eval_split:str):

    """
    
        Abe & Long, 1999: Associative Reinforcement Learning using Linear Probabilistic Concepts
    
    """
    observed_asr = {
        'train':0,
        eval_split:0
    }

    weight_log = {
        'train':np.zeros((T,n)),
        eval_split: np.zeros((T_prime-T,n))
    }

    X = context_vectors
    
    rng = np.random.default_rng(seed=cfg.seed)

    num_passes = cfg.attack.num_passes

    # Train loop
    for t in range(T):
        X_t = X['train'][t] #(n,d)
        y_t = X_t@w # (n,)
        y_t = y_t.clip(0,1) # clip prediction as per the pseudocode
        g_t = np.argmax(y_t)
        p_t = np.zeros(n)

        for i in range(n):
            if i == g_t: continue
            else: p_t[i] = 1/(n+4*kappa*(alpha-alpha**2)*(y_t[g_t] - y_t[i]))
        
        p_t[g_t] = 1 - p_t.sum()

        weight_log['train'][t,:] = p_t
        
        a_t = rng.choice(n, p=p_t)
        r_ta = 1 - expert_data['train'][t,a_t]

        w = w+alpha*(r_ta-X_t[a_t]@w)*X_t[a_t]
        
        observed_asr['train']+=r_ta
    
    rng = np.random.default_rng(seed=cfg.seed)
    # Test Loop
    for t in range(T_prime-T):
        X_t = X[eval_split][t] #(n,d)
        y_t = X_t@w # (n,)
        y_t = y_t.clip(0,1) # clip prediction as per the pseudocode
        g_t = np.argmax(y_t)
        p_t = np.zeros(n)

        for i in range(n):
            if i == g_t: continue
            else: p_t[i] = 1/(n+4*kappa*(alpha-alpha**2)*(y_t[g_t] - y_t[i]))
        
        p_t[g_t] = 1 - p_t.sum()

        weight_log[eval_split][t,:] = p_t
        
        if num_passes == 1:
            a_t = rng.choice(n, p=p_t)
            r_ta_learn = 1 - expert_data[eval_split][t,a_t]
        else:
            actions = rng.choice(n,p=p_t,size=num_passes,replace=False)
            a_t = actions[0]
            r_ta_learn = 1 - expert_data[eval_split][t,a_t]
            r_ta = compute_kshot_success_from_rewards(1-expert_data[eval_split][t,:], actions)

        if cfg.attack.continual:
            w = w+alpha*(r_ta_learn-X_t[a_t]@w)*X_t[a_t]
        
        if num_passes == 1:
            observed_asr[eval_split]+=r_ta_learn
        else:
            observed_asr[eval_split]+=r_ta
    
    if T>0:
        observed_asr['train']/=T
    observed_asr[eval_split]/=(T_prime-T)

    return dict(
        data=data,
        expert_data=expert_data,
        observed_asr=observed_asr,
        weight_log=weight_log
    )

def learn_weights_linucb(cfg:config.main.ProjectConfig, data:dict[str,pd.DataFrame], expert_data:dict[str,np.ndarray], T:int, T_prime:int,n:int, alpha:np.float64, context_vectors:dict[str,np.ndarray], w:np.ndarray, eval_split:str):

    """
    
        Li, Chu, Langford & Schapire, 2010: A Contextual-Bandit Approach to Personalized News Article Recommendation
    
    """

    observed_asr = {
        'train':0,
        eval_split:0
    }

    weight_log = {
        'train':np.zeros((T,n)),
        eval_split: np.zeros((T_prime-T,n))
    }
    
    X = context_vectors
    d = X[eval_split].shape[-1]
    A = np.array([np.eye(d) for _ in range(n)]) # n,d,d
    b = np.zeros((n,d))

    L_factor = []
    lower_flags = []

    for a in range(n):
        L, lower_flag = sp.linalg.cho_factor(A[a], lower=True)
        L_factor.append(L)
        lower_flags.append(lower_flag)
    
    L_factor = np.array(L_factor)
    lower_flags = np.array(lower_flags)

    num_passes = cfg.attack.num_passes

    # Train loop
    for t in range(T):
        X_t = X['train'][t] # n,d
        p_t = np.zeros(n)

        for a in range(n):
            theta = sp.linalg.cho_solve((L_factor[a], lower_flags[a]), b[a]) # Solves A@theta = b => theta = A_inv@b

            w = sp.linalg.cho_solve((L_factor[a], lower_flags[a]), X_t[a]) # Solves A@w = x => w = A_inv@x

            p_t[a] = np.dot(theta, X_t[a]) + alpha*np.sqrt(np.dot(X_t[a], w)) # theta.T@x_t + alpha*sqrt(x.T@A_inv@x)
        
        weight_log['train'][t,:] = p_t
        
        a_t = np.argmax(p_t)
        r_ta = 1 - expert_data['train'][t,a_t]

        A[a_t]+= np.outer(X_t[a_t], X_t[a_t])
        b[a_t]+= r_ta*X_t[a_t]
        L_factor[a_t], lower_flags[a_t] = sp.linalg.cho_factor(A[a_t], lower=True)
        
        observed_asr['train']+=r_ta
    
    # Test loop
    for t in range(T_prime-T):
        X_t = X[eval_split][t] # n,d
        p_t = np.zeros(n)

        for a in range(n):
            theta = sp.linalg.cho_solve((L_factor[a], lower_flags[a]), b[a]) # Solves A@theta = b => theta = A_inv@b

            w = sp.linalg.cho_solve((L_factor[a], lower_flags[a]), X_t[a]) # Solves A@w = x => w = A_inv@x

            p_t[a] = np.dot(theta, X_t[a]) + alpha*np.sqrt(np.dot(X_t[a], w)) # theta.T@x_t + alpha*sqrt(x.T@A_inv@x)
        
        weight_log[eval_split][t,:] = p_t
        
        if num_passes == 1:
            a_t = np.argmax(p_t)
            r_ta_learn = 1 - expert_data[eval_split][t,a_t]
        else:
            actions = np.argsort(p_t)[-num_passes:][::-1]
            a_t = actions[0]
            r_ta_learn = 1 - expert_data[eval_split][t,a_t]
            r_ta = compute_kshot_success_from_rewards(1 - expert_data[eval_split][t,:], actions)

        if cfg.attack.continual:
            A[a_t]+= np.outer(X_t[a_t], X_t[a_t])
            b[a_t]+= r_ta_learn*X_t[a_t]
            L_factor[a_t], lower_flags[a_t] = sp.linalg.cho_factor(A[a_t], lower=True)
        
        if num_passes == 1:
            observed_asr[eval_split]+=r_ta_learn
        else:
            observed_asr[eval_split]+=r_ta
    
    if T>0:
        observed_asr['train']/=T
    observed_asr[eval_split]/=(T_prime-T)

    return dict(
        data=data,
        expert_data=expert_data,
        observed_asr=observed_asr,
        weight_log=weight_log
    )

def learn_weights_square_cb(cfg:config.main.ProjectConfig, data:dict[str,pd.DataFrame], expert_data:dict[str,np.ndarray],T:int, T_prime:int,n:int,context_vectors:dict[str,np.ndarray], sq_alg:BaseRegressionOracle, Lambda:np.float64, mu:np.float64,eval_split:str):

    """
    
        Foster & Rakhlin, 2020: Beyond UCB: Optimal and Efficient Contextual Bandits with Regression Oracles
    
    """

    observed_asr = {
        'train':0,
        eval_split:0
    }

    weight_log = {
        'train':np.zeros((T,n)),
        eval_split: np.zeros((T_prime-T,n))
    }

    X = context_vectors

    rng = np.random.default_rng(seed=cfg.seed)

    num_passes = cfg.attack.num_passes

    # Train Loop
    for t in range(T):
        y_hat = np.zeros(n)
        for i in range(n):
            x_ti = X['train'][t,i,:]
            y_hat[i] = sq_alg.predict(x_ti)
        
        _a_t = np.argmin(y_hat)

        p_t = np.zeros(n)

        for i in range(n):
            if i == _a_t: continue
            p_t[i] = 1/(mu + Lambda*(y_hat[i] - y_hat[_a_t]))
        
        p_t[_a_t] = 1 - p_t.sum()

        weight_log['train'][t,:] = p_t
        
        a_t = rng.choice(n,p=p_t)
        y_t = expert_data['train'][t,a_t]
        r_ta = 1-expert_data['train'][t,a_t]

        sq_alg.update(X['train'][t,a_t,:], y_t)
        
        observed_asr['train']+=r_ta
    
    # Test loop
    for t in range(T_prime-T):
        y_hat = np.zeros(n)
        for i in range(n):
            x_ti = X[eval_split][t,i,:]
            y_hat[i] = sq_alg.predict(x_ti)
        
        _a_t = np.argmin(y_hat)

        p_t = np.zeros(n)

        for i in range(n):
            if i == _a_t: continue
            p_t[i] = 1/(mu + Lambda*(y_hat[i] - y_hat[_a_t]))
        
        p_t[_a_t] = 1 - p_t.sum()

        weight_log[eval_split][t,:] = p_t
        
        if num_passes == 1:
            a_t = rng.choice(n,p=p_t)
            y_t = expert_data[eval_split][t,a_t]
            r_ta = 1-expert_data[eval_split][t,a_t]
        else:
            actions = rng.choice(n,p=p_t, size=num_passes,replace=False)
            a_t = actions[0] # learning action
            y_t = expert_data[eval_split][t,a_t]
            r_ta = compute_kshot_success_from_rewards(1-expert_data[eval_split][t,:], actions)

        if cfg.attack.continual:
            sq_alg.update(X[eval_split][t,a_t,:], y_t)
        
        observed_asr[eval_split]+=r_ta
    
    if T>0:
        observed_asr['train']/=T
    observed_asr[eval_split]/=(T_prime-T)

    return dict(
        data=data,
        expert_data=expert_data,
        observed_asr=observed_asr,
        weight_log=weight_log
    )

def learn_weights_uniform_priors(cfg:config.main.ProjectConfig, data:dict[str,pd.DataFrame], expert_data:dict[str,np.ndarray], T:int, T_prime:int, n:int, eval_split:str):

    """

        Uniform Prior Baseline: This algorithm models a naive attacker that does not borrow priors from an online learning algorithm
    
    """

    observed_asr = {
        'train': 0,
        eval_split:0
    }

    weight_log = {
        'train': np.zeros((T,n)),
        eval_split: np.zeros((T_prime-T,n))
    }

    num_passes = cfg.attack.num_passes
    
    w = np.full(n, 1/n)
    
    rng = np.random.default_rng(seed=cfg.seed)

    # Train Loop (This is only used for the regret computation.)
    for t in range(T):
        reward_t = 1 - expert_data['train'][t,:]
        p_t = w

        weight_log['train'][t,:] = p_t

        i_t = rng.choice(n,p=p_t)

        r_it = reward_t[i_t]

        observed_asr['train']+=r_it

    rng = np.random.default_rng(seed=cfg.seed)
    # Test loop only
    for t in tqdm(range(T_prime-T), desc='Eval Rounds:'):
        reward_t = 1-expert_data[eval_split][t,:]
        p_t = w

        weight_log[eval_split][t,:] = p_t
        
        if num_passes == 1:
            i_t = rng.choice(n, p=p_t)

            r_it = reward_t[i_t]
        else:
            actions = rng.choice(n,p=p_t, size=num_passes, replace=False)
            r_it = compute_kshot_success_from_rewards(reward_t, actions)
        
        observed_asr[eval_split]+=r_it
    
    if T > 0:
        observed_asr['train']/=T
    observed_asr[eval_split]/=(T_prime-T)

    return dict(
        data=data,
        expert_data=expert_data,
        observed_asr=observed_asr,
        weight_log=weight_log
    )

def learn_weights_bcbf(cfg:config.main.ProjectConfig, data:dict[str,pd.DataFrame], expert_data:dict[str, np.ndarray], T:int, T_prime:int, n:int, eval_split:str):

    """
        Budget Constrained Brute Force (BCBF) baseline: This algorithm models an attacker that has the same budget of O(T) inference rounds with the target model, but uses the brute force approach rather than the learning approach, i.e., evaluates all n jailbreaks on a random sample of size (T/n) of the input data. 
    
    """

    observed_asr = {
        'train':0,
        eval_split:0
    }

    sample_size = T//n
    test_asr = []
    rng = np.random.default_rng(cfg.seed)
    idx = rng.choice(len(data['train']), size=sample_size, replace=False)
    sampled_train_data =  data['train'].iloc[idx]
    sampled_expert = expert_data['train'][idx]

    train_rewards = 1 - sampled_expert # (T/n,n) matrix of rewards
    average_asr:np.ndarray = train_rewards.mean(axis=0) # (n,)

    a = average_asr.argmax()
    observed_asr['train'] = average_asr[a]

    # Test Loop
    test_rewards = 1 - expert_data[eval_split]
    avg_test_asr = test_rewards.mean(axis=0)

    observed_asr[eval_split] = avg_test_asr[a]

    return dict(
        data=data,
        expert_data=expert_data,
        observed_asr=observed_asr,
        weight_log=None
    )

def learn_weights_thompson_sampling(cfg:config.main.ProjectConfig, data:dict[str, pd.DataFrame], expert_data:dict[str, np.ndarray], T:int, T_prime:int, n:int,alpha:np.ndarray, beta:np.ndarray, eval_split:str):

    """
    Thompson, 1933: On the Likelihood that One Unknown Probability Exceeds Another in View of the Evidence of Two Samples

    """

    observed_asr = {
        'train':0,
        eval_split:0
    }

    weight_log = {
        'train': np.zeros((T,n)),
        eval_split:np.zeros((T_prime-T,n))
    }

    rng = np.random.default_rng(seed=cfg.seed)

    tr_exp_data = expert_data['train']
    te_exp_data = expert_data[eval_split]

    num_passes = cfg.attack.num_passes

    #Train Loop
    for t in range(T):
        reward_t = 1-tr_exp_data[t,:]

        theta = rng.beta(alpha, beta)

        weight_log['train'][t,:] = theta

        a_t = np.argmax(theta)
        r_t = reward_t[a_t]

        alpha[a_t]+= r_t
        beta[a_t]+=1-r_t

        observed_asr['train']+=r_t
    
    rng = np.random.default_rng(seed=cfg.seed)

    # Test loop
    for t in range(T_prime-T):
        reward_t = 1-te_exp_data[t,:]

        theta = rng.beta(alpha, beta)

        weight_log[eval_split][t,:] = theta

        if num_passes == 1:
            a_t = np.argmax(theta)

            r_t_learn = reward_t[a_t]
        else:
            actions = np.argsort(theta)[-num_passes:][::-1]
            a_t = actions[0]
            r_t_learn = reward_t[a_t]
            r_t = compute_kshot_success_from_rewards(reward_t, actions)

        if cfg.attack.continual:
            alpha[a_t]+=r_t_learn
            beta[a_t]+=1-r_t_learn
        
        if num_passes == 1:
            observed_asr[eval_split]+=r_t_learn
        else:
            observed_asr[eval_split]+=r_t
    
    if T>0:
        observed_asr['train']/=T

    observed_asr[eval_split]/=(T_prime-T)

    return dict(
        data=data,
        expert_data=expert_data,
        observed_asr=observed_asr,
        weight_log=weight_log
    )