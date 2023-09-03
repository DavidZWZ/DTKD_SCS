# @Time   : 2020/7/21
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE:
# @Time   : 2021/7/9, 2020/9/17, 2020/8/31, 2021/2/20, 2021/3/1
# @Author : Yupeng Hou, Yushuo Chen, Kaiyuan Li, Haoran Cheng, Jiawei Guan
# @Email  : houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn, tsotfsk@outlook.com, chenghaoran29@foxmail.com, guanjw@ruc.edu.cn

"""
recbole.data.utils
########################
"""

import copy
import importlib
import os
import pickle 
import pdb

from recbole.data.dataloader import *
from recbole.sampler import KGSampler, Sampler, RepeatableSampler
from recbole.utils import ModelType, ensure_dir, get_local_time, set_color
from recbole.utils.argument_list import dataset_arguments


def create_dataset(config):
    """Create dataset according to :attr:`config['model']` and :attr:`config['MODEL_TYPE']`.
    If :attr:`config['dataset_save_path']` file exists and
    its :attr:`config` of dataset is equal to current :attr:`config` of dataset.
    It will return the saved dataset in :attr:`config['dataset_save_path']`.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Returns:
        Dataset: Constructed dataset.
    """
    dataset_module = importlib.import_module('recbole.data.dataset')
    if hasattr(dataset_module, config['model'] + 'Dataset'):
        dataset_class = getattr(dataset_module, config['model'] + 'Dataset')
    else:
        model_type = config['MODEL_TYPE']
        type2class = {
            ModelType.GENERAL: 'Dataset',
            ModelType.SEQUENTIAL: 'SequentialDataset',
            ModelType.CONTEXT: 'Dataset',
            ModelType.KNOWLEDGE: 'KnowledgeBasedDataset',
            ModelType.TRADITIONAL: 'Dataset',
            ModelType.DECISIONTREE: 'Dataset',
        }
        dataset_class = getattr(dataset_module, type2class[model_type]) # get Dataset class

    default_file = os.path.join(config['checkpoint_dir'], f'{config["dataset"]}-{dataset_class.__name__}.pth')
    file = config['dataset_save_path'] or default_file
    if os.path.exists(file):
        with open(file, 'rb') as f:
            dataset = pickle.load(f)
        dataset_args_unchanged = True
        for arg in dataset_arguments + ['seed', 'repeatable']:
            if config[arg] != dataset.config[arg]:
                dataset_args_unchanged = False
                break
        if dataset_args_unchanged:
            logger = getLogger()
            logger.info(set_color('Load filtered dataset from', 'pink') + f': [{file}]')
            return dataset

    dataset = dataset_class(config)
    if config['save_dataset']:
        dataset.save()
    return dataset


def save_split_dataloaders_cs(config, dataloaders):
    """Save split dataloaders.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataloaders (tuple of AbstractDataLoader): The split dataloaders.
    """
    ensure_dir(config['checkpoint_dir'])
    save_path = config['checkpoint_dir']
    saved_dataloaders_file = config["dataset"]
    dataloader_file = 'dataloader.pth'
    if config['model'] == 'BPR': # added for pariwise loss of BPR model
        file_path = os.path.join(save_path, saved_dataloaders_file, 'dataloader_BPR.pth')    
    elif config['sample_train_only']:# added for sample train only kd
        file_path = os.path.join(save_path, saved_dataloaders_file, 'dataloader_sample_train.pth')   
    else:
        file_path = os.path.join(save_path, saved_dataloaders_file, dataloader_file)
    logger = getLogger()
    logger.info(set_color('Saving split dataloaders into', 'pink') + f': [{file_path}]')
    with open(file_path, 'wb') as f:
        pickle.dump(dataloaders, f)


def load_split_dataloaders_cs(config):
    """Load split cold-start dataloaders if saved dataloaders exist and
    their :attr:`config` of dataset are the same as current :attr:`config` of dataset.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Returns:
        dataloaders (tuple of AbstractDataLoader or None): The split dataloaders.
    """
    if config['model'] == 'BPR': # added for pariwise loss of BPR model
        default_file = os.path.join(config['checkpoint_dir'], config["dataset"], 'dataloader_BPR.pth')
    elif config['sample_train_only']:
        default_file = os.path.join(config['checkpoint_dir'], config["dataset"], 'dataloader_sample_train.pth')       
    else:
        default_file = os.path.join(config['checkpoint_dir'], config["dataset"], 'dataloader.pth')
    dataloaders_save_path = config['dataloaders_save_path'] or default_file

    if not os.path.exists(dataloaders_save_path):
        return None
    with open(dataloaders_save_path, 'rb') as f:
        ww_train_data, ww_valid_data, ww_test_data, cw_test_data, wc_test_data, cc_test_data = pickle.load(f)
    for arg in dataset_arguments + ['seed', 'repeatable', 'eval_args']:
        if config[arg] != ww_train_data.config[arg]:
            return None
    ww_train_data.update_config(config)
    ww_valid_data.update_config(config)
    ww_test_data.update_config(config)
    cw_test_data.update_config(config)
    wc_test_data.update_config(config)
    cc_test_data.update_config(config)

    logger = getLogger()
    logger.info(set_color('Load split dataloaders from', 'pink') + f': [{dataloaders_save_path}]')
    return ww_train_data, ww_valid_data, ww_test_data, cw_test_data, wc_test_data, cc_test_data


def save_split_dataloaders(config, dataloaders):
    """Save split dataloaders.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataloaders (tuple of AbstractDataLoader): The split dataloaders.
    """
    ensure_dir(config['checkpoint_dir'])
    save_path = config['checkpoint_dir']
    saved_dataloaders_file = f'{config["dataset"]}-for-{config["model"]}-dataloader.pth'
    file_path = os.path.join(save_path, saved_dataloaders_file)
    logger = getLogger()
    logger.info(set_color('Saving split dataloaders into', 'pink') + f': [{file_path}]')
    with open(file_path, 'wb') as f:
        pickle.dump(dataloaders, f)


def load_split_dataloaders(config):
    """Load split dataloaders if saved dataloaders exist and
    their :attr:`config` of dataset are the same as current :attr:`config` of dataset.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Returns:
        dataloaders (tuple of AbstractDataLoader or None): The split dataloaders.
    """

    default_file = os.path.join(config['checkpoint_dir'], f'{config["dataset"]}-for-{config["model"]}-dataloader.pth')
    dataloaders_save_path = config['dataloaders_save_path'] or default_file
    if not os.path.exists(dataloaders_save_path):
        return None
    with open(dataloaders_save_path, 'rb') as f:
        train_data, valid_data, test_data = pickle.load(f)
    for arg in dataset_arguments + ['seed', 'repeatable', 'eval_args']:
        if config[arg] != train_data.config[arg]:
            return None
    train_data.update_config(config)
    valid_data.update_config(config)
    test_data.update_config(config)
    logger = getLogger()
    logger.info(set_color('Load split dataloaders from', 'pink') + f': [{dataloaders_save_path}]')
    return train_data, valid_data, test_data


def data_preparation_cs(config, dataset):
    """Split the dataset by :attr:`config['eval_args']` and create training, validation and test dataloader.

    Note:
        If we can load split dataloaders by :meth:`load_split_dataloaders`, we will not create new split dataloaders.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataset (Dataset): An instance object of Dataset, which contains all interaction records.

    Returns:
        tuple:
            - train_data (AbstractDataLoader): The dataloader for warm user warm item
            - test_data (AbstractDataLoader): The dataloader for warm user new item

    """
    dataloaders = load_split_dataloaders_cs(config)
    if dataloaders is not None:
        ww_train_data, ww_valid_data, ww_test_data, cw_test_data, wc_test_data, cc_test_data = dataloaders
    else:
        built_datasets = dataset.build_cs()
        ww_train_dataset, ww_valid_dataset, ww_test_dataset, cw_test_dataset, wc_test_dataset, cc_test_dataset = built_datasets
        train_sampler, valid_sampler, test_sampler, cw_test_sampler, wc_test_sampler, cc_test_sampler = create_samplers_cs(config, dataset, built_datasets)

        ww_train_data = get_dataloader(config, 'train')(config, ww_train_dataset, train_sampler, shuffle=True)
        ww_valid_data = get_dataloader(config, 'evaluation')(config, ww_valid_dataset, valid_sampler, shuffle=False)
        ww_test_data = get_dataloader(config, 'evaluation')(config, ww_test_dataset, test_sampler, shuffle=False)
        cw_test_data = get_dataloader(config, 'evaluation')(config, cw_test_dataset, cw_test_sampler, shuffle=False)
        wc_test_data = get_dataloader(config, 'evaluation')(config, wc_test_dataset, wc_test_sampler, shuffle=False)
        cc_test_data = get_dataloader(config, 'evaluation')(config, cc_test_dataset, cc_test_sampler, shuffle=False)
    
        if config['save_dataloaders']:
            save_split_dataloaders_cs(config, dataloaders=(ww_train_data, ww_valid_data, ww_test_data, cw_test_data, wc_test_data, cc_test_data))

        logger = getLogger()
        logger.info(
        set_color('[Training]: ', 'pink') + set_color('train_batch_size', 'cyan') + ' = ' +
        set_color(f'[{config["train_batch_size"]}]', 'yellow') + set_color(' negative sampling', 'cyan') + ': ' +
        set_color(f'[{config["neg_sampling"]}]', 'yellow')
    )
        logger.info(
        set_color('[Evaluation]: ', 'pink') + set_color('eval_batch_size', 'cyan') + ' = ' +
        set_color(f'[{config["eval_batch_size"]}]', 'yellow') + set_color(' eval_args', 'cyan') + ': ' +
        set_color(f'[{config["eval_args"]}]', 'yellow')
    )
    
    return ww_train_data, ww_valid_data, ww_test_data, cw_test_data, wc_test_data, cc_test_data

def data_preparation(config, dataset):
    """Split the dataset by :attr:`config['eval_args']` and create training, validation and test dataloader.

    Note:
        If we can load split dataloaders by :meth:`load_split_dataloaders`, we will not create new split dataloaders.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataset (Dataset): An instance object of Dataset, which contains all interaction records.

    Returns:
        tuple:
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    dataloaders = load_split_dataloaders(config)
    if dataloaders is not None:
        train_data, valid_data, test_data = dataloaders
    else:
        model_type = config['MODEL_TYPE']
        built_datasets = dataset.build()
        train_dataset, valid_dataset, test_dataset = built_datasets
        train_sampler, valid_sampler, test_sampler = create_samplers(config, dataset, built_datasets)

        if model_type != ModelType.KNOWLEDGE:
            train_data = get_dataloader(config, 'train')(config, train_dataset, train_sampler, shuffle=True)
        else:
            kg_sampler = KGSampler(dataset, config['train_neg_sample_args']['distribution'])
            train_data = get_dataloader(config, 'train')(config, train_dataset, train_sampler, kg_sampler, shuffle=True)

        valid_data = get_dataloader(config, 'evaluation')(config, valid_dataset, valid_sampler, shuffle=False)
        test_data = get_dataloader(config, 'evaluation')(config, test_dataset, test_sampler, shuffle=False)
        if config['save_dataloaders']:
            save_split_dataloaders(config, dataloaders=(train_data, valid_data, test_data))
    logger = getLogger()
    logger.info(
        set_color('[Training]: ', 'pink') + set_color('train_batch_size', 'cyan') + ' = ' +
        set_color(f'[{config["train_batch_size"]}]', 'yellow') + set_color(' negative sampling', 'cyan') + ': ' +
        set_color(f'[{config["neg_sampling"]}]', 'yellow')
    )
    logger.info(
        set_color('[Evaluation]: ', 'pink') + set_color('eval_batch_size', 'cyan') + ' = ' +
        set_color(f'[{config["eval_batch_size"]}]', 'yellow') + set_color(' eval_args', 'cyan') + ': ' +
        set_color(f'[{config["eval_args"]}]', 'yellow')
    )
    return train_data, valid_data, test_data


def get_dataloader(config, phase):
    """Return a dataloader class according to :attr:`config` and :attr:`phase`.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        phase (str): The stage of dataloader. It can only take two values: 'train' or 'evaluation'.

    Returns:
        type: The dataloader class that meets the requirements in :attr:`config` and :attr:`phase`.
    """
    register_table = {
        "MultiDAE": _get_AE_dataloader,
        "MultiVAE": _get_AE_dataloader,
        'MacridVAE': _get_AE_dataloader,
        'CDAE': _get_AE_dataloader,
        'ENMF': _get_AE_dataloader,
        'RaCT': _get_AE_dataloader,
        'RecVAE': _get_AE_dataloader,
    }

    if config['model'] in register_table:
        return register_table[config['model']](config, phase)

    model_type = config['MODEL_TYPE']
    if phase == 'train':
        if model_type != ModelType.KNOWLEDGE:
            return TrainDataLoader
        else:
            return KnowledgeBasedDataLoader
    else:
        eval_strategy = config['eval_neg_sample_args']['strategy']
        if eval_strategy in {'none', 'by', 'label'}:
            return NegSampleEvalDataLoader
        elif eval_strategy == 'full':
            return FullSortEvalDataLoader


def _get_AE_dataloader(config, phase):
    """Customized function for VAE models to get correct dataloader class.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        phase (str): The stage of dataloader. It can only take two values: 'train' or 'evaluation'.

    Returns:
        type: The dataloader class that meets the requirements in :attr:`config` and :attr:`phase`.
    """
    if phase == 'train':
        return UserDataLoader
    else:
        eval_strategy = config['eval_neg_sample_args']['strategy']
        if eval_strategy in {'none', 'by'}:
            return NegSampleEvalDataLoader
        elif eval_strategy == 'full':
            return FullSortEvalDataLoader


def create_samplers_cs(config, dataset, built_datasets):
    """Create sampler for cold_start evaluation

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataset (Dataset): An instance object of Dataset, which contains all interaction records.
        built_datasets (list of Dataset): A list of split Dataset, which contains dataset for
            ww_train, cw_test, wc_test, cc_test

    Returns:
        tuple:
            - ww_train_sampler (AbstractSampler): The sampler for training.
            - cw_test_sampler (AbstractSampler): The sampler for testing.
            - wc_test_sampler (AbstractSampler): The sampler for testiîng.
            - cc_test_sampler (AbstractSampler): The sampler for testing.
    """

    phases = ['train', 'valid_ww', 'test_ww', 'test_cw', 'test_wc', 'test_cc']
    train_neg_sample_args = config['train_neg_sample_args']
    eval_neg_sample_args = config['eval_neg_sample_args']
    sampler = None
    train_sampler, test_sampler = None, [None]*5

    if train_neg_sample_args['strategy'] != 'none':
        if not config['repeatable']:
            sampler = Sampler(phases, built_datasets, train_neg_sample_args['distribution'],\
                 eval_mode = 'cs', positive_len=config['positive_len'])
        else:
            sampler = RepeatableSampler(phases, dataset, train_neg_sample_args['distribution'])
        train_sampler = sampler.set_phase('train')

    if eval_neg_sample_args['strategy'] != 'none':
        if sampler is None:
            if not config['repeatable']:
                sampler = Sampler(phases, built_datasets, eval_neg_sample_args['distribution'], \
                    eval_mode = 'cs', positive_len=config['positive_len'])
            else:
                sampler = RepeatableSampler(phases, dataset, eval_neg_sample_args['distribution'])
        else:
            sampler.set_distribution(eval_neg_sample_args['distribution'])
        for i, phase in enumerate(phases[1:]):
            test_sampler[i] = sampler.set_phase(phase)

    return train_sampler, test_sampler[0], test_sampler[1], test_sampler[2], test_sampler[3], test_sampler[4]


def create_samplers(config, dataset, built_datasets):
    """Create sampler for training, validation and testing.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataset (Dataset): An instance object of Dataset, which contains all interaction records.
        built_datasets (list of Dataset): A list of split Dataset, which contains dataset for
            training, validation and testing.

    Returns:
        tuple:
            - train_sampler (AbstractSampler): The sampler for training.
            - valid_sampler (AbstractSampler): The sampler for validation.
            - test_sampler (AbstractSampler): The sampler for testing.
    """
    phases = ['train', 'valid', 'test']
    train_neg_sample_args = config['train_neg_sample_args']
    eval_neg_sample_args = config['eval_neg_sample_args']
    sampler = None
    train_sampler, valid_sampler, test_sampler = None, None, None

    if train_neg_sample_args['strategy'] != 'none':
        if not config['repeatable']:
            sampler = Sampler(phases, built_datasets, train_neg_sample_args['distribution'])
        else:
            sampler = RepeatableSampler(phases, dataset, train_neg_sample_args['distribution'])
        train_sampler = sampler.set_phase('train')

    if eval_neg_sample_args['strategy'] != 'none':
        if sampler is None:
            if not config['repeatable']:
                sampler = Sampler(phases, built_datasets, eval_neg_sample_args['distribution'])
            else:
                sampler = RepeatableSampler(phases, dataset, eval_neg_sample_args['distribution'])
        else:
            sampler.set_distribution(eval_neg_sample_args['distribution'])
        valid_sampler = sampler.set_phase('valid')
        test_sampler = sampler.set_phase('test')

    return train_sampler, valid_sampler, test_sampler
