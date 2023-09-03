import argparse
import os
from recbole.quick_start import run_recbole, run_recbole_cs, run_recbole_kd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='WideDeep', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='ml-1m', help='name of datasets')
    parser.add_argument('-r', type=str, default='kd', help='running method type')

    args, _ = parser.parse_known_args()
    config_file_list = ['config.yaml']

    if args.r == 'cs':
        run_recbole_cs(model=args.model, dataset=args.dataset, config_file_list=config_file_list)
    elif args.r == 'kd':
        run_recbole_kd(dataset=args.dataset, config_file_list=config_file_list)
    elif args.r == 'rec':
        run_recbole(model=args.model, dataset=args.dataset, config_file_list=config_file_list)
    else:
        raise Exception('Implementation does not exist')