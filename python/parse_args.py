import argparse
import textwrap

def parse_argument(sys_argv):
    """Parses arguments from command line.

    Args:
        sys_argv: the list of arguments (strings) from command line.

    Returns:
        A struct whose member corresponds to the required (optional) variable.
        For example,
        ```
        args = parse_argument(['main.py' '--input', 'a.txt', '--num', '10'])
        args.input       # 'a.txt'
        args.num         # 10
        ```
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='Deep learning models for image classification')

    # Training parameters

    parser.add_argument(
        '--train-data', type=str, required=True,
        help=textwrap.dedent(
            '''
            Train data path
            '''))
    parser.add_argument(
        '--val-data', type=str, required=True,
        help=textwrap.dedent(
            '''
            Validation data path
            '''))


    parser.add_argument(
        '--num_dataload_worker', type=int,
        default=4,
        help='number of dataloader workers'
    )

    parser.add_argument(
        '--optimizer', type=str, default='name=sgd, momentum=0.0',
        help=textwrap.dedent(
            '''
            Supported optimizer OPTIM (specified by name=OPTIM):
              1) "sgd"
              2) "adagrad"
              3) "adam"
              4) "adamw"
            Syntax:
              "name=OPTIM, ..." # followed by optimizer-based arguments
            '''))



    parser.add_argument(
        '--max-steps', type=int, default=None,
        help='The number of outer iterations')
    parser.add_argument(
        '--warmup-ratio', type=float, default=0.03,
        help='The number of outer iterations')
    parser.add_argument(
        '--epoch', type=int, default=3,
        help='The number of epochs')
    parser.add_argument(
        '--global_batch_size', type=int, required=True, default=1,
        help='Global batch size')

    parser.add_argument(
        '--micro_batch_size', type=int, required=True, default=1,
        help='Micro batch size')

    parser.add_argument(
        '--val_batch_size', type=int, default=4,
        help='Val micro batch size')

    # Model parameters
    parser.add_argument(
        '--model-type', type=str, default='Llama', choices=['Llama', 'Gemma2', 'Qwen2'], 
        help=textwrap.dedent(
            '''
            Supported models:
              * Llama
              * Gemma2
              * Qwen2
            '''))
    
    # Model parameters
    parser.add_argument(
        '--model', type=str)
    
    parser.add_argument(
        '--tokenizer-name', type=str, default=None,)
    
    parser.add_argument(
        '--bf16', action='store_true',
        help='Whether use bf16 training')
    
    parser.add_argument(
        '--lora', action='store_true',
        help='Whether use LoRA training')
    parser.add_argument(
        '--lisa', action='store_true',
        help='Whether use Lisa training')
    parser.add_argument(
        '--lisa-step', type=int, default=20,
        help='Lisa step')
    # Test parameters
    parser.add_argument(
        '--eval_frequency', type=int, default=1000000,
        help='Evaluate every {eval_frequency} inner iterations')

    # Debug parameters
    parser.add_argument(
        '--seed', type=int, default=23,
        help='Random seed that controls the pseudo-random behavior')
    parser.add_argument(
        '--pseudo_random', const=True, default=False, nargs='?',
        help='A global option to make all random operations deterministic')
    parser.add_argument(
        '--logging_conf_file', default='conf/common.log_conf', type=str,
        help='The configuration file for logging')
    parser.add_argument(
        '--use_wandb', action='store_true',
        help='Turn on wandb logging',
    )
    parser.add_argument(
        '--norm', type=float, default=None,
        help='Turn on l2 norm',
    )
    parser.add_argument(
        '--wandb_project', type=str, default='forgetting',
        help='Project name for wandb',
    )
    parser.add_argument(
        '--wandb_run_name', type=str, default=None,
        help='Run name for wandb',
    )
    parser.add_argument(
        '--save_dir', default=None, type=str,
        help='Where to save the model')

    parser.add_argument(
        '--diff_norm', action='store_true',
        help='use norm of difference with base model')

    parser.add_argument(
        '--lmflow-format', action='store_true',
        help='use lmflow format data')

    
    parser.add_argument(
        '--max-length', type=int,
        default=256,
        help='number of partitions'
    )
    parser.add_argument(
        '--response_loss_only', action='store_true',
        help='Only use loss of response'
    )
    parser.add_argument(
        '--sharegpt_format', action='store_true',
        help='Only use loss of response'
    )

    parser.add_argument(
        '--pretrain', action='store_true',
        help='Pretrain mode'
    )
    # Parses from commandline
    args = parser.parse_args(sys_argv[1:])

    return args


def parse_args_dict(args_string):
    """Parses strings like 'type=sgd, momentum=0.0, nesterov=True'.

    Returns:
        A dict mapping names to their values, e.g.
        {
            'type': 'sgd',
            'momentum': 0.0,
            'nesterov': True,
        }
    """
    args_string = args_string.replace(' ', '').replace(',', '\n')
    args_string = "[model_args]\n" + args_string

    import configparser
    import io
    import ast

    # Uses config parser to parse the input string, the result dict has values
    # with str types
    buf = io.StringIO(args_string)
    config = configparser.ConfigParser()
    config.read_file(buf)

    args_dict = { s: dict(config.items(s)) for s in config.sections() }

    # Use .liter_eval to decide types of the value, e.g. 0.0 -> float, True ->
    # bool
    new_args_dict = {}
    for k, v in args_dict['model_args'].items():
        try:
            tmp_v = ast.literal_eval(v)
            new_args_dict[k] = tmp_v
        except:
            new_args_dict[k] = v
            continue

    return new_args_dict