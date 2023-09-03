"""
recbole.quick_start
########################
"""
import logging
from logging import getLogger

import torch
import pickle
import pdb
import os

from recbole.config import Config
from recbole.data import create_dataset, data_preparation, data_preparation_cs, save_split_dataloaders, load_split_dataloaders
from recbole.utils import init_logger, get_model, get_trainer, init_seed, set_color, ResultLogger
from recbole.evaluator import Collector

def run_recbole_cs(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):
    r""" A fast running api for strict cold-start, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    # configurations initialization
    saved_model_file = None
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)
    res_looger = ResultLogger(config)

    # dataset filtering
    dataset = create_dataset(config)
    # dataset splitting
    ww_train_data, ww_valid_data, ww_test_data, cw_test_data, wc_test_data, cc_test_data = \
                                        data_preparation_cs(config, dataset)
    # model loading and initialization
    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, ww_train_data.dataset).to(config['device'])
    logger.info(model)
    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        ww_train_data, ww_valid_data, validate_mode='cs', load_model_file=saved_model_file, \
            saved=saved, show_progress=config['show_progress']
    )

    # model evaluation
    print("warm user warm item")
    test_result = trainer.evaluate_cs(ww_test_data, 0, load_best_model=saved, model_file=saved_model_file,\
    show_progress=config['show_progress'])
    logger.info(set_color('test result', 'yellow') + f': {test_result}')
    res_looger.log_result(result = test_result, eval_state = "ww")

    print("cold user warm item")
    test_result = trainer.evaluate_cs(cw_test_data, 1, load_best_model=saved, model_file=saved_model_file,\
    show_progress=config['show_progress'])
    logger.info(set_color('test result', 'yellow') + f': {test_result}')
    res_looger.log_result(result = test_result, eval_state = "cw")

    print("warm user cold item")
    test_result = trainer.evaluate_cs(wc_test_data, 2, load_best_model=saved, model_file=saved_model_file,\
    show_progress=config['show_progress'])
    logger.info(set_color('test result', 'yellow') + f': {test_result}')
    res_looger.log_result(result = test_result, eval_state = "wc")

    print("cold user cold item")
    test_result = trainer.evaluate_cs(cc_test_data, 3, load_best_model=saved, model_file=saved_model_file,\
    show_progress=config['show_progress'])
    logger.info(set_color('test result', 'yellow') + f': {test_result}')
    res_looger.log_result(result = test_result, eval_state = "cc")

    res_looger.write_result()
    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


def run_recbole_kd(dataset=None, config_file_list=None, config_dict=None, saved=True):
    r""" A fast running api for Dual-Teacher Knowledge Distillation, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    # configurations initialization
    distill_mode = 'sf'
    teacher_cf_model = "DirectAUT" 
    teacher_con_model = "LinearT" 
    student_model = 'MLPS' 

    config_con = Config(model=teacher_con_model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    config_cf = Config(model=teacher_cf_model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    config_s = Config(model=student_model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    init_seed(config_con['seed'], config_con['reproducibility'])
    # logger initialization
    res_looger = ResultLogger(config_s)
    saved_con_model_file = config_s['saved_con_model_file']
    saved_cf_model_file = config_s['saved_cf_model_file']

    # dataset filtering
    dataset = create_dataset(config_con)
    # dataset splitting
    ww_train_data, ww_valid_data, ww_test_data, cw_test_data, wc_test_data, cc_test_data = \
                                        data_preparation_cs(config_con, dataset)

    teacher_con = Model(model=teacher_con_model, train_data=ww_train_data, config=config_con, saved=saved)
    teacher_cf = Model(model=teacher_cf_model, train_data=ww_train_data, config=config_cf, saved=saved)
    student = Model(model=student_model, train_data=ww_train_data, config=config_s, saved=saved)

    # Teacher model training
    best_valid_score, best_valid_result = teacher_con.trainer.fit(
        ww_train_data, validate_mode='cs', teacher_type="con", load_model_file=saved_con_model_file,
        saved=saved, show_progress=config_con['show_progress'])
    best_valid_score, best_valid_result = teacher_cf.trainer.fit(
        ww_train_data, validate_mode='cs', teacher_type="cf", load_model_file=saved_cf_model_file, 
        saved=saved, show_progress=config_con['show_progress'])

    # Student model training
    best_valid_score, best_valid_result = student.trainer.kd(ww_train_data, ww_valid_data, teacher_con.trainer, 
    teacher_cf.trainer, validate_mode = 'cs', distill_mode=distill_mode, saved=saved, show_progress=config_con['show_progress']
    )
    
    # model evaluation
    print("warm user warm item")
    test_result = student.trainer.evaluate_cs(ww_test_data, 0, load_best_model=saved, \
                                            show_progress=config_s['show_progress'])
    student.logger.info(set_color('test result', 'yellow') + f': {test_result}')
    res_looger.log_result(result = test_result, eval_state = "stu ww")

    print("cold user warm item")
    test_result = student.trainer.evaluate_cs(cw_test_data, 1, load_best_model=saved, \
                                            show_progress=config_s['show_progress'])
    student.logger.info(set_color('test result', 'yellow') + f': {test_result}')
    res_looger.log_result(result = test_result, eval_state = "stu cw")

    print("warm user cold item")
    test_result = student.trainer.evaluate_cs(wc_test_data, 2, load_best_model=saved, \
                                            show_progress=config_s['show_progress'])
    student.logger.info(set_color('test result', 'yellow') + f': {test_result}')
    res_looger.log_result(result = test_result, eval_state = "stu wc")

    print("cold user cold item")
    test_result = student.trainer.evaluate_cs(cc_test_data, 3, load_best_model=saved, \
                                            show_progress=config_s['show_progress'])
    student.logger.info(set_color('test result', 'yellow') + f': {test_result}')
    res_looger.log_result(result = test_result, eval_state = "stu cc")
    
    res_looger.write_result()
    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config_s['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }

class Model(object):
    """model init
    """
    def __init__(self, model=None, train_data=None, config=None, saved=True):

        # logger initialization
        init_logger(config)
        self.logger = getLogger()
        self.logger.info(config)

        # model loading and initialization
        init_seed(config['seed'], config['reproducibility'])
        self.model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
        self.logger.info(self.model)

        # trainer loading and initialization
        self.trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, self.model)


def run_recbole(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):
    """ A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    # configurations initialization
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)
    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)
    # model loading and initialization
    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    logger.info(model)
    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, validate_mode='rec', saved=saved, show_progress=config['show_progress']
    )
    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=config['show_progress'])
    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


def objective_function(config_dict=None, config_file_list=None, saved=True):
    r""" The default objective_function used in HyperTuning

    Args:
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """

    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config['seed'], config['reproducibility'])
    logging.basicConfig(level=logging.ERROR)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, verbose=False, saved=saved)
    test_result = trainer.evaluate(test_data, load_best_model=saved)

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


def load_data_and_model(model_file):
    r"""Load filtered dataset, split dataloaders and saved model.

    Args:
        model_file (str): The path of saved model file.

    Returns:
        tuple:
            - config (Config): An instance object of Config, which record parameter information in :attr:`model_file`.
            - model (AbstractRecommender): The model load from :attr:`model_file`.
            - dataset (Dataset): The filtered dataset.
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    checkpoint = torch.load(model_file)
    config = checkpoint['config']
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    model.load_state_dict(checkpoint['state_dict'])
    model.load_other_parameter(checkpoint.get('other_parameter'))

    return config, model, dataset, train_data, valid_data, test_data
