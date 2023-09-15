from pytorch_lightning.logging import LightningLoggerBase
from tensorboardX import SummaryWriter
from pathlib import Path

import sys
import os


class Logger(LightningLoggerBase):
    """Class for logging output."""

    def __init__(self, kwargs):
        self.exp_dir = os.path.join(kwargs.get("save_path"),
                                    kwargs.get("exp_name"))
        self.log_path = Path(self.exp_dir, kwargs.get("log_name") + ".txt")
        self.log_file = self.log_path.open('w+')

        self.tb_log_dir = os.path.join(self.exp_dir, "tb")
        self.summary_writer = SummaryWriter(log_dir=self.tb_log_dir)

        self.results_dir = kwargs.get("results_dir")
        if self.results_dir is not None:
            self.metrics_path = os.path.join(self.results_dir, "scores.txt")
        self.metrics_file = self.metrics_path.open('w+')

    def log(self, *args):
        self.log_stdout(*args)
        print(*args, file=self.log_file)
        self.log_file.flush()

    def log_metrics(self, metrics, step_num):
        for metric, value in metrics.items():
            msg = f'{metric}:\t{value}'
            if self.results_dir is not None:
                self.log_stdout(msg)
                print(msg, file=self.metrics_file)
                self.metrics_file.flush()
            else:
                self.log(f"[{msg}]")

    def log_stdout(self, *args):
        print(*args, file=sys.stdout)
        sys.stdout.flush()

    def close(self):
        self.log_file.close()

    def log_hyperparams(self, params):
        pass

    def log_scalars(self, scalar_dict, iterations, print_to_stdout=True):
        """Log all values in a dict as scalars to TensorBoard."""
        for k, v in scalar_dict.items():
            if print_to_stdout:
                self.log_stdout(f'[{k}: {v:.3g}]')
            k = k.replace('_', '/')  # Group in TensorBoard by phase
            self.summary_writer.add_scalar(k, v, iterations)
