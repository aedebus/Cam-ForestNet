import nni
import time
import logging
import json
import traceback
from main import train


def _cast_value(v):
    if v == "True":
        v = True
    elif v == "False":
        v = False
    elif v == "None":
        v = None
    return v


if __name__ == "__main__":
    try:
        params = nni.get_next_parameter()
        baseline_hp_path = params.pop("baseline_hp_path")
        logging.info(f"Load base parameter from [baseline_hp_path]")
        with open(baseline_hp_path) as f:
            base_params = json.load(f)
        base_params = {k: _cast_value(v['_value'][0])
                       for k, v in base_params.items()}
        logging.info("Base Params:")
        logging.info(base_params)

        logging.info("Experiment Params:")
        logging.info(params)

        params = {k: _cast_value(v) for k, v in params.items()}
        base_params.update(params)
        base_params['exp_name'] = "nni" + str(time.time())
        logging.info("Final Params:")
        logging.info(base_params)

        train(**base_params)
    except RuntimeError as re:
        if 'out of memory' in str(re):
            time.sleep(600)
            params['batch_size'] = int(0.5 * params['batch_size'])
            train(**params)
        else:
            traceback.print_exc()
            nni.report_final_result(-1)
    except Exception as e:
        traceback.print_exc()
        nni.report_final_result(-2)
