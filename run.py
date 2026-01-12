# run.py 수정

import sys
from logging import getLogger
import yaml
import argparse
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, init_seed
from recbole.trainer import Trainer
from model import * 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='flowcf.yaml')
    
    # [추가된 옵션]
    parser.add_argument('--prior', type=str, default=None, help='gaussian or bernoulli')
    parser.add_argument('--act', type=str, default=None, help='gelu or leakyrelu')
    
    args = parser.parse_args()

    # 1. Config 로드
    with open(args.config, 'r', encoding='utf-8') as file:
        yaml_config = yaml.safe_load(file)
    
    # 2. 옵션 덮어쓰기 (커맨드라인 인자가 있으면 우선순위)
    config_dict = {}
    if args.prior:
        config_dict['prior_type'] = args.prior
    if args.act:
        config_dict['act_func'] = args.act
        
    # 모델명 가져오기
    model_name = yaml_config.get('model', 'FlowCF')

    # 3. RecBole Config 초기화 (config_dict 추가)
    config = Config(model=locals()[model_name], config_file_list=[args.config], config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    
    # ... (이후 동일) ...
    init_logger(config)
    logger = getLogger()
    
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = locals()[config['model']](config, train_data.dataset).to(config['device'])
    logger.info(model)

    trainer = Trainer(config, model)

    # 저장 파일명에 옵션 붙이기 (구분을 위해)
    run_name = f"FlowCF_{config['prior_type']}_{config['act_func']}"
    config['checkpoint_dir'] = 'saved/' 
    
    # 학습
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, show_progress=config["show_progress"], saved=True
    )
    
    # 저장된 모델 파일명 출력 (자동으로 시간붙어서 저장됨)
    print(f"\n[Training Finished] Model settings: {config['prior_type']} + {config['act_func']}")