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

# [핵심] 두 모델 클래스를 모두 가져옵니다.
from model.flowcf import FlowCF
try:
    from model.diffcf import DiffCF
except ImportError:
    DiffCF = None  # DiffCF 파일이 없을 경우를 대비

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
    # 모델 설정값 읽기
    if hasattr(model, 'prior_type'):
        dist_name = model.prior_type.capitalize() 
    else:
        dist_name = "Gaussian" 

    if hasattr(model, 'act_func'):
        act_name = model.act_func.upper() 
        if act_name == 'LEAKYRELU':
            act_name = 'LeakyReLU'
    else:
        act_name = "GELU"
    
    return dist_name, act_name

def print_custom_table(results, count, model_name, dist_name, act_name, dataset_name="MovieLens-1M"):
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
    
    print(f"| {model_name:^10} | {dist_name:^10} | {act_name:^10} | {recall_str:^22} | {ndcg_str:^22} |")
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
    dist_name, act_name = get_model_settings(model)
    model_name = model.__class__.__name__ # 현재 사용 중인 클래스 이름 (DiffCF or FlowCF)
    
    movie_field = dataset.uid_field 
    user_field = dataset.iid_field 
    max_k = max(k_list)

    # [앙상블] 성능 향상을 위해 여러 번 추론 후 평균 (DiffCF일 때 효과적)
    # 속도가 중요하다면 1로 설정, 성능이 중요하다면 5 추천
    n_repeat = 5 if model_name == 'DiffCF' else 1

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

            # 앙상블 적용
            final_scores = None
            for _ in range(n_repeat):
                scores = model.predict_cold_item(int(internal_mid)) 
                if final_scores is None:
                    final_scores = scores
                else:
                    final_scores += scores
            
            final_scores = final_scores / n_repeat
            final_scores = final_scores.view(-1)
            
            _, top_indices = torch.topk(final_scores, max_k)
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
        print_custom_table(results, count, model_name, dist_name, act_name)
    else:
        print("평가된 아이템이 없습니다.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='flowcf.yaml')
    parser.add_argument('--test_file', type=str, default='dataset/ML1M/BPR_cv/BPR_cv.test.inter')
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()

    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint)
    
    # 저장된 Config 로드
    saved_config = checkpoint['config']
    
    # [수정] .get() 제거 및 안전한 접근 (배치 사이즈 자동 조절)
    if 'eval_batch_size' in saved_config and saved_config['eval_batch_size'] < 32768:
        print(f"Warning: 저장된 eval_batch_size({saved_config['eval_batch_size']})가 작습니다. 65536으로 임시 변경합니다.")
        saved_config['eval_batch_size'] = 65536

    init_seed(saved_config['seed'], saved_config['reproducibility'])
    dataset = create_dataset(saved_config)
    
    # [핵심 수정] .get() 제거: Config 객체는 딕셔너리가 아니므로 직접 접근 또는 try-except 사용
    try:
        model_name = saved_config['model']
    except KeyError:
        model_name = 'FlowCF' # 예전 버전 호환성
    
    print(f"Detected Model Type: {model_name}")

    if model_name == 'DiffCF':
        if DiffCF is None:
            raise ImportError("model/diffcf.py 파일을 찾을 수 없습니다.")
        model = DiffCF(saved_config, dataset).to(saved_config['device'])
    else:
        model = FlowCF(saved_config, dataset).to(saved_config['device'])

    model.load_state_dict(checkpoint['state_dict'])

    evaluate_cold_start(model, dataset, args.test_file, k_list=[10, 20])