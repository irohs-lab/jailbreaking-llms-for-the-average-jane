import itertools
import hashlib
import json


def generate_param_grid(cfg):

    combos = []

    target_models = cfg.ol_val_sweep.target_models

    for group in cfg.ol_val_sweep.groups:

        param_space = group.params

        keys = list(param_space.keys())
        values = list(param_space.values())

        for vals in itertools.product(*values):

            params = dict(zip(keys, vals))

            for target_model in target_models:

                combo_dict = dict(
                    ol_algo=cfg.ol_scheme.name,
                    params=params,
                    target_model=target_model
                )

                combo_id = hashlib.md5(
                    json.dumps(combo_dict, sort_keys=True).encode()
                ).hexdigest()

                combos.append(
                    dict(
                        params=params,
                        target_model=target_model,
                        id=combo_id
                    )
                )

    print(f"Total sweep jobs: {len(combos)}")

    return dict(param_grid=combos)