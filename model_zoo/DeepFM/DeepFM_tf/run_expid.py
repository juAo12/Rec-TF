import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))

import logging
from datetime import datetime
import sys
from yactr.utils import load_config, set_logger, print_to_json, print_to_list
from yactr.features import FeatureMap
from yactr.TFscr.tf_utils import seed_everything
from yactr.TFscr.dataloaders import TFRecordDataLoader
import tensorflow as tf
import src as model_zoo
import gc
import argparse
from pathlib import Path

if __name__ == '__main__':
    ''' Usage: python run_expid.py --config {config_dir} --expid {experiment_id} --gpu {gpu_device_id}
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/', help='The config directory.')
    parser.add_argument('--expid', type=str, default='DeepFM_test', help='The experiment id to run.')
    parser.add_argument('--gpu', type=int, default=-1, help='The gpu index, -1 for cpu')
    args = vars(parser.parse_args())

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if args['gpu'] >= 0:
        tf.config.set_visible_devices(gpus[args['gpu']], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[args['gpu']], True)
    experiment_id = args['expid']
    params = load_config(args['config'], experiment_id)
    set_logger(params)
    logging.info("Params: " + print_to_json(params))
    seed_everything(seed=params['seed'])

    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_map_json = os.path.join(data_dir, "feature_map.json")
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(feature_map_json, params)
    logging.info("Feature specs: " + print_to_json(feature_map.features))

    model_class = getattr(model_zoo, params['model'])
    model = model_class(feature_map, **params)

    train_gen, valid_gen = TFRecordDataLoader(feature_map, stage='train', **params).make_iterator()
    model.fit(train_gen, validation_data=valid_gen, **params)

    logging.info('****** Validation evaluation ******')
    valid_result = model.evaluate(valid_gen)
    del train_gen, valid_gen
    gc.collect()

    logging.info('******** Test evaluation ********')
    test_gen = TFRecordDataLoader(feature_map, stage='test', **params).make_iterator()
    test_result = {}
    if test_gen:
        test_result = model.evaluate(test_gen)

    result_filename = Path(args['config']).name.replace(".yaml", "") + '.csv'
    with open(result_filename, 'a+') as fw:
        fw.write(' {},[command] python {},[exp_id] {},[dataset_id] {},[train] {},[val] {},[test] {}\n' \
                 .format(datetime.now().strftime('%Y%m%d-%H%M%S'),
                         ' '.join(sys.argv), experiment_id, params['dataset_id'],
                         "N.A.", print_to_list(valid_result), print_to_list(test_result)))
