import config.main
from src.utils import regression_oracles
from src.utils.dr import reduce_dimensionality
import pandas as pd
import numpy as np
import os
import torch


def get_context_vectors(cfg:config.main.ProjectConfig, data:dict[str, pd.DataFrame]):

    model_key = cfg.ol_scheme.context_vector_model.split('/')[-1].replace('-','_')

    if cfg.ol_scheme.use_matryoshka and cfg.ol_scheme.truncate_dim is not None:
        model_key+=f"_matryoshka_{cfg.ol_scheme.truncate_dim}"

    root = os.path.join(
        cfg.paths.data_dir,
        cfg.paths.sent_emb_data_dir,
        model_key
    )

    context_vectors = {}
    jailbreaks = [jb if jb is not None else "baseline" for jb in cfg.eval.jailbreaks]

    for split, split_df in data.items():
        if len(split_df) == 0: continue
        context_vectors[split] = []
        for jb in jailbreaks:
            cv_file = os.path.join(
                root,
                split,
                f"{jb}.pt"
            )

            cv_cache = torch.load(cv_file)
            phashes = cv_cache['prompt_hashes']
            embeddings = cv_cache['embeddings']
            hash_emb_map = {
                phash : emb.cpu().to(torch.float32).numpy() 
                for phash,emb in zip(phashes,embeddings)
            }
            
            ordered_embeddings = np.stack([hash_emb_map[phash] for phash in split_df['prompt_hash'].tolist()])

            context_vectors[split].append(ordered_embeddings)
        context_vectors[split] = np.stack(context_vectors[split],axis=1) # T,n,d

        if cfg.ol_scheme.dr_solver is not None and cfg.ol_scheme.context_dim is not None:
            context_vectors[split] = reduce_dimensionality(cfg.ol_scheme.dr_solver,context_vectors[split], cfg.ol_scheme.context_dim, cfg.seed) # T,n,d'

        if cfg.ol_scheme.normalize_cv:
            norms = np.linalg.norm(context_vectors[split], axis=-1, keepdims=True)
            context_vectors[split] = context_vectors[split]/np.where(norms==0,1,norms)
    
    return context_vectors

def check_dimensions(context_vectors:dict[str,np.ndarray], data:dict[str, pd.DataFrame], jailbreaks:list[str]):
    
    for split, split_df in data.items():
        if len(split_df) == 0: continue
        assert context_vectors[split].shape[0] == len(split_df)
        assert context_vectors[split].shape[1] == len(jailbreaks)
    
    dims = [v.shape[-1] for v in context_vectors.values()]
    assert all(d == dims[0] for d in dims), f"Dimension mismatch found: {dims}"

def get_oracle_kwargs(reg_oracle:str, **oracle_args)->dict:

    regressor_params = {}

    if reg_oracle == 'VovkOnlineRegressionOracle':
        regressor_params['num_rounds'] = oracle_args['num_rounds']
        return regressor_params

def get_ol_kwargs(cfg:config.main.ProjectConfig, data:dict[str,pd.DataFrame], expert_data:dict[str,np.ndarray], eval_split="test"):

    ol_kwargs = {}
    T = len(expert_data['train'])
    T_prime = T + len(expert_data[eval_split])
    n = expert_data['train'].shape[1]

    ol_kwargs['T_prime'] = T + len(expert_data[eval_split])
    ol_kwargs['T'] = T
    ol_kwargs['n'] = n
    ol_kwargs['eval_split'] = eval_split

    if cfg.ol_scheme.name == 'randomized_weighted_majority':
        _w = np.ones(n)
        if cfg.attack.continual:
            lr = cfg.ol_scheme.lr if cfg.ol_scheme.lr else np.sqrt(2*np.log(n)/T_prime)
        else:
            lr = cfg.ol_scheme.lr if cfg.ol_scheme.lr else np.sqrt(2*np.log(n)/T)
        ol_kwargs['_w'] = _w
        ol_kwargs['lr'] = lr
    elif cfg.ol_scheme.name == "exp3":
        _w = np.ones(n)
        if cfg.attack.continual:
            lr = cfg.ol_scheme.lr if cfg.ol_scheme.lr else np.min([1, np.sqrt((n*np.log(n))/((np.e-1)*T_prime))])
        else:
            lr = cfg.ol_scheme.lr if cfg.ol_scheme.lr else np.min([1, np.sqrt((n*np.log(n))/((np.e-1)*T))])
        ol_kwargs['_w'] = _w
        ol_kwargs['lr'] = lr
    elif cfg.ol_scheme.name == "linear_cb":
        if not cfg.attack.continual:
            ol_kwargs['kappa'] = (T**(3/4)) * (n**(1/2)) * 0.5
            ol_kwargs['alpha'] = 1/(T**0.5)
        else:
            T_prime = T + len(expert_data[eval_split])
            ol_kwargs['kappa'] = (T_prime**(3/4)) * (n**(1/2)) * 0.5
            ol_kwargs['alpha'] = 1/(T_prime**0.5)
        context_vectors = get_context_vectors(cfg,data)

        check_dimensions(context_vectors, data, cfg.eval.jailbreaks)

        ol_kwargs['context_vectors'] = context_vectors
        d = context_vectors[eval_split].shape[-1]
        ol_kwargs['w'] = np.zeros(d)
    elif cfg.ol_scheme.name == 'linucb':
        fp = cfg.ol_scheme.failure_prob
        ol_kwargs['alpha'] = 1 + np.sqrt(np.log(2/fp)*0.5)

        context_vectors = get_context_vectors(cfg,data)

        check_dimensions(context_vectors, data, cfg.eval.jailbreaks)

        ol_kwargs['context_vectors'] = context_vectors
        d = context_vectors[eval_split].shape[-1]
        ol_kwargs['w'] = np.zeros(d)
    elif cfg.ol_scheme.name == "square_cb":
        context_vectors = get_context_vectors(cfg,data)
        check_dimensions(context_vectors, data, cfg.eval.jailbreaks)

        ol_kwargs['context_vectors'] = context_vectors
        d = context_vectors[eval_split].shape[-1]
        online_regressor_class = getattr(regression_oracles, cfg.ol_scheme.regression_oracle)

        regressor_params = dict(cfg.ol_scheme.online_regressor_params)
        regressor_params.update({'dim':d})

        T_prime = T + len(expert_data[eval_split])
        regressor_params.update(
            get_oracle_kwargs(
                cfg.ol_scheme.regression_oracle,
                **{
                    'num_rounds': T if not cfg.attack.continual else T_prime
                }
            )
        )

        ol_kwargs['sq_alg'] = online_regressor_class(**regressor_params)

        reg_sq = ol_kwargs['sq_alg'].regret_bound(T if not cfg.attack.continual else T_prime)

        mu = n
        ol_kwargs['mu'] = mu
        delta = cfg.ol_scheme.failure_prob

        if cfg.attack.continual:
            ol_kwargs['Lambda'] = np.sqrt( (mu*T_prime)/(reg_sq + np.log(2/delta)))
        else:
            ol_kwargs['Lambda'] = np.sqrt( (mu*T)/(reg_sq + np.log(2/delta)))
    elif cfg.ol_scheme.name == "thompson_sampling":
        ol_kwargs['alpha'] = np.ones(n)
        ol_kwargs['beta'] = np.ones(n)

    elif cfg.ol_scheme.name=="uniform_priors":
        pass
    elif cfg.ol_scheme.name == "bcbf":
        pass

    else:
        raise NotImplementedError(f"get_ol_kwargs() does not support cfg.ol_scheme = {cfg.ol_scheme.name} yet")
    
    return dict(data=data, expert_data=expert_data, **ol_kwargs)