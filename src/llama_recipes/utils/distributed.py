from torch import distributed as torch_distributed
import os


def is_rank_0() -> bool:
    return torch_distributed.is_initialized() and torch_distributed.get_rank() == 0


def print_rank_0(message) -> None:
    if torch_distributed.is_initialized() and torch_distributed.get_rank() == 0:
        print(message, flush=True)


def set_mpi_env() -> None:
    global_rank = int(os.getenv("OMPI_COMM_WORLD_RANK", 0))
    local_rank = int(os.getenv("OMPI_COMM_WORLD_LOCAL_RANK", 0))
    world_size = int(os.getenv("OMPI_COMM_WORLD_SIZE", 1))

    os.environ["RANK"] = str(global_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)


def get_rank() -> int:
    if torch_distributed.is_initialized():
        return torch_distributed.get_rank()
    else:
        return 0


def get_world_size() -> int:
    if torch_distributed.is_initialized():
        return torch_distributed.get_world_size()
    else:
        return 1


def get_local_rank() -> int:
    if torch_distributed.is_initialized():
        return int(os.getenv("LOCAL_RANK", 0))
    else:
        return 0


def fsdp_auto_wrap_policy(model, transformer_layer_name):
    import functools

    from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy

    def lambda_policy_fn(module):
        if (
            len(list(module.named_children())) == 0
            and getattr(module, "weight", None) is not None
            and module.weight.requires_grad
        ):
            return True
        return False

    lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=(
            transformer_layer_name,  # type: ignore
        ),
    )

    auto_wrap_policy = functools.partial(_or_policy, policies=[lambda_policy, transformer_wrap_policy])
    return auto_wrap_policy
