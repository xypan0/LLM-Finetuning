import torch
import copy
import wandb
from accelerate.logging import get_logger
import time
from tqdm.auto import tqdm as original_tqdm
from functools import partial

from torch.distributed.fsdp import FullyShardedDataParallel

def make_tqdm(accelerator, list_data):
    tqdm = partial(original_tqdm, disable=not accelerator.is_local_main_process, position=0)
    return tqdm(list_data)

def get_optimizer(model_params, optimizer_args_dict):
    """Gets optimizer given a configuration.

    Args:
        model_params: same type as model.parameters().
        optimizer_args_dict: a dict mapping optimizer arguments to their values.
            Except one key called "name", which specifies the optimizer name.
    """
    name = optimizer_args_dict['name']
    new_optimizer_args_dict = copy.deepcopy(optimizer_args_dict)
    new_optimizer_args_dict.pop('name', None)

    if name == 'sgd':
        return torch.optim.SGD(model_params, **new_optimizer_args_dict)
    elif name == 'adagrad':
        return torch.optim.Adagrad(model_params, **new_optimizer_args_dict)
    elif name == 'adam':
        return torch.optim.Adam(model_params, **new_optimizer_args_dict)
    elif name == 'adamw':
        return torch.optim.AdamW(model_params, **new_optimizer_args_dict)
    else:
        raise ValueError(f'Optimizer "{name}" is not supported')
    
@torch.no_grad()
def evaluate(model, accelerator, data_loader, args):
    """Evaluates model performance on dataset `data_loader`."""
    torch_rng_state = torch.get_rng_state()

    model.eval()
    sum_loss = 0.

    for batch in data_loader:
        batch=batch.to(accelerator.device)
        x_batch=batch['input_ids']
        y_batch=batch['labels']
        attn_mask=batch['attention_mask']
        x_batch = x_batch# .cuda()
        y_batch = y_batch# .cuda()
        attn_mask = attn_mask# .cuda()

        # if args.norm:
            # loss, norm, res = model(input_ids=x_batch, labels=y_batch, attention_mask=attn_mask, return_dict=True)
        # else:
        loss = model(input_ids=x_batch, labels=y_batch, attention_mask=attn_mask, return_dict=True).loss

        sum_loss += accelerator.gather(loss).detach().cpu().mean()

    num_batch = len(data_loader)
    eval_loss = sum_loss / float(num_batch)

    # Resets the random state, we don't want evaluation to affect the
    # reproducibility of the main training process! However, even we use
    # separated dataloaders (even unshuffled), they can affect the random state
    # (https://github.com/pytorch/pytorch/issues/11062). So we have to reset
    # torch's random state.
    torch.set_rng_state(torch_rng_state)
    return eval_loss

def logging_stat_dict(stat_dict, prefix='', suffix='', use_wandb=False, accelerator=None):
    logger = get_logger('accelerator')
    stat_str_list = [f'{prefix}']
    for key, value in stat_dict.items():
        stat_str_list.append(f' {key} = {value},')
    stat_str_list.append(f'{suffix}')

    stat_str = ''.join(stat_str_list)
    logger.info(stat_str)

    if use_wandb and accelerator.is_main_process:
        wandb.log(stat_dict)

def evaluate_and_logging(model, global_step, start_time, args, accelerator, val_loader):
    if global_step < 0 or (global_step % args.eval_frequency == 0):
        # eval_val_loss, eval_val_acc = evaluate(model, accelerator, val_loader_for_eval, config.args)
        eval_loss = evaluate(model, accelerator, val_loader, args)

        prefix = f'At the beginning of i = {global_step}:'

        suffix = ' (evaluation mode of model)'

        stat_dict = {
            'validation loss': eval_loss,
            'step': global_step,
            'time': time.time() - start_time,
        }
        logging_stat_dict(stat_dict, prefix, suffix, args.use_wandb, accelerator)

def save_model_(accelerator, accelerate_model, tokenizer, save_dir, norm=False, **kargs):

    accelerator.wait_for_everyone()
    accelerator.print(f"saving model at {save_dir} ...")
    # unwrapped_model = accelerator.unwrap_model(accelerate_model)
    # print(unwrapped_model)
    # if norm:
        # unwrapped_model = unwrapped_model.targetModule
    # print(accelerator.get_state_dict(unwrapped_model))
    with FullyShardedDataParallel.summon_full_params(accelerate_model, with_grads=False):
        if norm:
            unwrapped_model = accelerate_model.targetModule
        print(unwrapped_model)
        unwrapped_model.save_pretrained(
            save_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=unwrapped_model.state_dict(),
            # state_dict=accelerator.get_state_dict(unwrapped_model),
            # state_dict=accelerator.get_state_dict(accelerate_model),
            max_shard_size="2GB"
        )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        tokenizer.save_pretrained(save_dir)

def save_model(accelerator, accelerate_model, tokenizer, save_dir, **kargs):

    accelerator.wait_for_everyone()
    accelerator.print(f"saving model at {save_dir} ...")
    unwrapped_model = accelerator.unwrap_model(accelerate_model)
    unwrapped_model.save_pretrained(
        save_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(accelerate_model),
        # max_shard_size="2GB"
    )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        tokenizer.save_pretrained(save_dir)

def logging_stat_dict(stat_dict, prefix='', suffix='', use_wandb=False, accelerator=None):
    logger = get_logger('accelerator')
    stat_str_list = [f'{prefix}']
    for key, value in stat_dict.items():
        stat_str_list.append(f' {key} = {value},')
    stat_str_list.append(f'{suffix}')

    stat_str = ''.join(stat_str_list)
    logger.info(stat_str)

    if use_wandb and accelerator.is_main_process:
        wandb.log(stat_dict)