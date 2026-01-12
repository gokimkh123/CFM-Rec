import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys

# 현재 경로 추가
sys.path.append(os.getcwd())

from recbole.config import Config
from recbole.data import create_dataset
from recbole.utils import init_seed
from model.flowcf import FlowCF

def compute_recall_ndcg(top_k_indices, ground_truth_internal_ids, k):
    hits = 0
    sum_r = 0.0
    gt_set = set(ground_truth_internal_ids)
    
    for i, idx in enumerate(top_k_indices):
        if idx in gt_set:
            hits += 1
            sum_r += 1.0 / np.log2(i + 2)

    recall = hits / len(gt_set) if len(gt_set) > 0 else 0.0
    idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(gt_set), k))])
    ndcg = sum_r / idcg if idcg > 0 else 0.0
    
    return recall, ndcg

def get_model_settings(model):
    """
    모델이 기억하고 있는 설정값(prior_type, act_func)을 읽어옵니다.
    """
    # 1. 초기 분포 확인
    if hasattr(model, 'prior_type'):
        dist_name = model.prior_type.capitalize() # gaussian -> Gaussian
    else:
        dist_name = "Gaussian" # 기본값

    # 2. 활성화 함수 확인
    if hasattr(model, 'act_func'):
        act_name = model.act_func.upper() # gelu -> GELU
        if act_name == 'LEAKYRELU':
            act_name = 'LeakyReLU'
    else:
        # 혹시 속성이 없으면 모듈 뒤져서 찾기 (구형 호환성)
        act_name = "Unknown"
        for module in model.modules():
            if isinstance(module, nn.GELU):
                act_name = "GELU"
                break
            elif isinstance(module, nn.LeakyReLU):
                act_name = "LeakyReLU"
                break
    
    return dist_name, act_name

def print_custom_table(results, count, dist_name, act_name, dataset_name="MovieLens-1M"):
    r10 = results.get(10, {'recall': 0})['recall'] / count if count > 0 else 0
    r20 = results.get(20, {'recall': 0})['recall'] / count if count > 0 else 0
    n10 = results.get(10, {'ndcg': 0})['ndcg'] / count if count > 0 else 0
    n20 = results.get(20, {'ndcg': 0})['ndcg'] / count if count > 0 else 0
    
    print("\n")
    print(f"{' ':35}{dataset_name}")
    
    border = "+" + "-"*12 + "+" + "-"*12 + "+" + "-"*12 + "+" + "-"*24 + "+" + "-"*24 + "+"
    print(border)
    print(f"| {'Methods':^10} | {'Prior':^10} | {'Act.Func':^10} | {'Recall @10 / @20':^22} | {'NDCG @10 / @20':^22} |")
    print("+" + "="*12 + "+" + "="*12 + "+" + "="*12 + "+" + "="*24 + "+" + "="*24 + "+")
    
    recall_str = f"{r10:.4f} / {r20:.4f}"
    ndcg_str = f"{n10:.4f} / {n20:.4f}"
    
    print(f"| {'FlowCF':^10} | {dist_name:^10} | {act_name:^10} | {recall_str:^22} | {ndcg_str:^22} |")
    print(border)
    print("\n")

def evaluate_cold_start(model, dataset, test_file_path, k_list=[10, 20]):
    print(f"\n[평가 시작] Cold-Start Item 평가 (File: {test_file_path})")
    
    if not os.path.exists(test_file_path):
        print(f"Error: 파일을 찾을 수 없습니다: {test_file_path}")
        return

    df = pd.read_csv(test_file_path, sep='\t')
    df.columns = [col.split(':')[0] for col in df.columns]
    
    test_items = df.groupby('mid')['uid'].apply(list).to_dict()
    print(f"총 테스트 아이템(Cold Items) 수: {len(test_items)}")
    
    results = {k: {'recall': 0.0, 'ndcg': 0.0} for k in k_list}
    count = 0
    
    model.eval()
    
    # 모델 설정 정보 가져오기
    dist_name, act_name = get_model_settings(model)
    
    movie_field = dataset.uid_field 
    user_field = dataset.iid_field 
    max_k = max(k_list)

    with torch.no_grad():
        for i, (mid_raw, uids_raw) in enumerate(test_items.items()):
            try:
                internal_mid = dataset.token2id(movie_field, str(mid_raw))
            except (ValueError, KeyError):
                continue

            gt_internal_uids = []
            for uid in uids_raw:
                try:
                    gt_internal_uids.append(dataset.token2id(user_field, str(uid)))
                except:
                    pass
            
            if not gt_internal_uids:
                continue

            scores = model.predict_cold_item(int(internal_mid)) 
            scores = scores.view(-1)
            
            _, top_indices = torch.topk(scores, max_k)
            top_indices = top_indices.cpu().numpy()
            
            for k in k_list:
                current_top = top_indices[:k]
                rec, ndcg = compute_recall_ndcg(current_top, gt_internal_uids, k)
                results[k]['recall'] += rec
                results[k]['ndcg'] += ndcg
            
            count += 1
            
            if (i+1) % 100 == 0:
                print(f"Processed {i+1}/{len(test_items)} items...")

    if count > 0:
        print_custom_table(results, count, dist_name, act_name)
    else:
        print("평가된 아이템이 없습니다.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # config 인자는 이제 필수가 아니지만 호환성을 위해 남겨둠
    parser.add_argument('--config', type=str, default='flowcf.yaml')
    parser.add_argument('--test_file', type=str, default='dataset/ML1M/BPR_cv/BPR_cv.test.inter')
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()

    # [핵심 수정] 1. 체크포인트를 먼저 로드합니다.
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint)
    
    # [핵심 수정] 2. 체크포인트 안에 저장된 '학습 당시의 Config'를 꺼냅니다.
    # RecBole은 저장할 때 config 객체 전체를 저장합니다.
    saved_config = checkpoint['config']
    
    # 3. 저장된 Config로 데이터셋과 모델을 초기화합니다.
    # 이렇게 해야 모델이 학습할 때 썼던 prior_type, act_func를 기억합니다.
    init_seed(saved_config['seed'], saved_config['reproducibility'])
    dataset = create_dataset(saved_config)
    
    model = FlowCF(saved_config, dataset).to(saved_config['device'])
    model.load_state_dict(checkpoint['state_dict'])

    evaluate_cold_start(model, dataset, args.test_file, k_list=[10, 20])