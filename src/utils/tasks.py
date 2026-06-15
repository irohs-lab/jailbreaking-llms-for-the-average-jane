import inspect
from collections.abc import Callable

PIPELINE_ABORT = object()
PIPELINE_RESTART = object()
PIPELINE_SKIP = object()

def validation(callable: Callable):
    """ Decorator to silently handle assertion errors and terminate the pipeline on failure.
        Useful for validation functions. """

    def safe_callable(*args, **kwargs):
        try:
            return callable(*args, **kwargs)
        except AssertionError as exc:
            print(" ", exc)
            return PIPELINE_ABORT

    safe_callable.__name__ = callable.__name__
    safe_callable.__signature__ = inspect.signature(callable)
    return safe_callable

def wait_for_user(action: str, is_complete: Callable[..., bool], on_complete: Callable[..., None]):
    """ Waits for a manual action, then proceeds with the pipeline run upon completion. """

    def wait_for_user_impl(config):
        already_completed = True
        while not is_complete(config):
            already_completed = False
            try:
                print("  Manual intervention required:", action)
                print("  Press a key when complete ... ", end='', flush=True)
                input()
            except KeyboardInterrupt:
                if not is_complete(config):
                    print("  incomplete.")
                break
        if not already_completed and is_complete(config):
            on_complete(config)
        elif not already_completed:
            return PIPELINE_ABORT

    wait_for_user_impl.__name__ = wait_for_user.__name__
    return wait_for_user_impl

def conditional(condition: Callable[..., bool]):
    """ Decorator to run a function or callable only if a condition is satisfied. """
    condition_signature = inspect.signature(condition)

    def make_conditional(callable: Callable):
        def run_if_condition_successful(*args, **kwargs):
            _kwargs = {
                key: value for key, value in kwargs.items()
                if key in condition_signature.parameters
            }
            if condition(*args, **_kwargs):
                return callable(*args, **kwargs)
            else:
                return PIPELINE_SKIP

        run_if_condition_successful.__name__ = callable.__name__
        run_if_condition_successful.__signature__ = inspect.signature(callable)
        return run_if_condition_successful
    return make_conditional

class Pipeline:
    def __init__(self, *jobs, **common_kwargs):
        self.jobs = jobs
        self.common_kwargs = common_kwargs
        self.job_kwargs = {
            job.__name__: list(inspect.signature(job).parameters.keys())
            for job in jobs
        }

        accelerator = self.common_kwargs.get('accelerator', None)
        self.print = accelerator.print if accelerator else print
        self.accelerator = accelerator

    def run(self, preserve_intermediates=True):
        _do_restart = False
        net_result = { **self.common_kwargs }

        for job in self.jobs:
            parameters = {
                attr: value for attr, value in net_result.items()
                if attr in self.job_kwargs[job.__name__]
            }

            if not preserve_intermediates:
                for attr in self.job_kwargs[job.__name__]:
                    if attr not in self.common_kwargs: del net_result[attr]

            self.print("> Running", job.__name__, "...")
            result = job(**parameters)
            if self.accelerator: self.accelerator.wait_for_everyone()

            # handle pipeline abort
            if isinstance(result, object) and result == PIPELINE_ABORT:
                break

            # handle pipeline restart
            if isinstance(result, object) and result == PIPELINE_RESTART:
                _do_restart = True
                break

            if isinstance(result, object) and result == PIPELINE_SKIP:
                self.print(" ", job.__name__, "skipped.")
            else:
                self.print(" ", job.__name__, "completed successfully.")
            self.print()

            if result is None: result = {}
            if not isinstance(result, dict):
                result = { job.__name__.split('_', maxsplit=1)[-1]: result }

            net_result = { **net_result, **result }

        if _do_restart:
            return self.run(preserve_intermediates)

        return {
            attr: value for attr, value in net_result.items()
            if attr not in self.common_kwargs
        }