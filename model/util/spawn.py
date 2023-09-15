import os
import argparse
from datetime import datetime
from subprocess import check_call

from constants import *


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--exp_tag',
        type=str,
        required=True,
        help='Tag to use in experiment names.'
    )

    parser.add_argument(
        '--models',
        type=str,
        default='ResNet101',
        help='Comma-separated list of models to run.'
    )

    parser.add_argument(
        '--segmentation',
        action="store_true",
        help='Whether to do segmentation.'
    )

    parser.add_argument(
        '--architectures',
        type=str,
        default='UNet',
        help='Comma-separated list of architectures to use.\
                Only used if --segmentation is passed, and\
                must be the same length as --models.'
    )

    parser.add_argument(
        '--loss_fns',
        type=str,
        default='CE',
        help='Comma-separated list of loss functions to use.\
                Only used if --segmentation is passed, and\
                must be the same length as --models.'
    )

    parser.add_argument(
        '--gammas',
        type=str,
        default='2',
        help='Comma-separated list of focal loss gamma values to use.\
                Only used if --segmentation is passed, and\
                must be the same length as --models.'
    )

    parser.add_argument(
        '--lrs',
        type=str,
        default='0.0001',
        help='Comma-separated list of lrs to use.'
    )

    parser.add_argument(
        '--extra_bands',
        type=str,
        default='',
        help='Comma-separated list of extra bands to include.\
                e.g. masked or masked,ir.'
    )

    parser.add_argument(
        '--augmentations',
        type=str,
        default='flip',
        help='Comma-separated list of augmentations to use.'
    )

    parser.add_argument(
        '--ckpt_metric',
        type=str,
        default='val_loss',
        help='Metric used to save checkpoints.'
    )

    parser.add_argument(
        '--gpus',
        type=str,
        default='[0,1]',
        help='GPUs to use.'
    )

    parser.add_argument(
        '--use_new_data',
        type=str,
        default='True',
        help='Whether to use newly downloaded data.'
    )


    parser.add_argument(
        '--spawn_type',
        type=str,
        choices=('stdout', 'shell', 'sbatch'),
        default='stdout',
        help='Whether to print commands to stdout (stdout), spawn \
                them on the current machine (shell), or spawn \
                them using sbatch (sbatch).'
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()

    models = args.models.split(',')
    if args.segmentation:
        architectures = args.architectures.split(',')
        loss_fns = args.loss_fns.split(',')
        gammas = args.gammas.split(',')
        if len(models) != len(architectures):
            raise ValueError("Must provide exactly one model " +
                             "for each architecture.")
        if len(models) != len(loss_fns):
            raise ValueError("Must provide exactly one loss fn " +
                             "for each model.")
        if len(models) != len(gammas):
            raise ValueError("Must provide exactly one gamma value " +
                             "for each model.")

    lrs = args.lrs.split(',')
    augmentations = args.augmentations.split(',')

    if args.extra_bands == '':
        bands = []
    else:
        bands = args.extra_bands.split(',')

    allowed_bands = ["masked", "ir"]
    for band in bands:
        if band not in allowed_bands:
            raise ValueError(f"Band {band} not supported.")

    today = datetime.today()
    month_day_str = f"{today.month:02}_{today.day:02}"
    max_epochs = 200
    patience = 100

    commands = []
    for i, model in enumerate(models):
        for lr in lrs:
            for aug in augmentations:

                hyperparam_suffix = f"{lr}_{model}_{aug}_{args.ckpt_metric}"
                if len(bands) > 0:
                    hyperparam_suffix += f"_{'_'.join(bands)}"

                exp_name = f"{month_day_str}_{args.exp_tag}_{hyperparam_suffix}"
                name = f'run_{hyperparam_suffix}'

                command = 'python main.py train'
                command += f' --dataset indonesia'
                command += f' --gpus {args.gpus}'
                command += f' --fixed_class_weights True'
                command += f' --ckpt_metric {args.ckpt_metric}'
                command += f' --max_epochs {max_epochs}'
                command += f' --patience {patience}'
                command += f' --model {model}'
                command += f' --augmentation {aug}'
                command += f' --use_new_data {args.use_new_data}'
                if args.segmentation:
                    command += f' --architecture {architectures[i]}'
                    command += f' --loss_fn {loss_fns[i]}'
                    command += f' --gamma {gammas[i]}'
                    command += ' --zoomed_regions False'
                    command += ' --segmentation'
                    exp_name += f'_{architectures[i]}_{loss_fns[i]}'
                    exp_name += f'_{gammas[i]}_segmentation'
                command += f' --lr {lr}'

                for band in bands:
                    command += f' --{band} True'

                command += f' --exp_name {exp_name}'

                commands.append((name, command))

    if args.spawn_type == 'stdout':
        # Print commands to stdout.
        for _, command in commands:
            print(command)

    elif args.spawn_type == 'shell':
        # Run commands sequentially on this machine.
        commands_str = ";".join([com for _, com in commands])
        print(commands_str)
        check_call(commands_str, shell=True)

    else:
        # Output the commands to a sbatch script then call sbatch.
        with open("util/sbatch_template.sh") as f:
            template = f.read()

        gpus_obj = eval(args.gpus)
        if isinstance(gpus_obj, list):
            num_gpus = len(gpus_obj)
        else:
            num_gpus = 1

        for name, command in commands:
            sbatch = template.replace("@COMMAND", command)
            sbatch = sbatch.replace("@NAME", name)

            sbatch = sbatch.replace("@GPUS", f"{num_gpus}")

            with open("spawn_tmp.sh", 'w') as f:
                f.write(sbatch)

            print(command)
            check_call("sbatch spawn_tmp.sh", shell=True)

        os.remove("spawn_tmp.sh")
