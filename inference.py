import argparse
import torch
import numpy as np
import os
import sys

# 경로 설정
sys.path.append(os.getcwd())

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed
from model.flowcf import FlowCF

def inference_cold_item(model, dataset, target_mid, k=10):
    """
    특정 영화(Mid)에 대해 좋아할 유저들을 추론
    """
    print(f"\n[추론 시작] Target Movie ID: {target_mid}")

    # 1. 모델이 학습한 Side Info 범위 내인지 확인
    if target_mid >= model.side_emb.shape[0]:
        print(f"Error: 입력한 Mid({target_mid})는 임베딩 파일 범위(0~{model.side_emb.shape[0]-1})를 벗어났습니다.")
        return

    # 2. Cold-Start 예측 실행
    # (내부적으로 Side Info를 가져와서 Noise와 결합해 유저 벡터 생성)
    with torch.no_grad():
        scores = model.predict_cold_item(int(target_mid))
    
    # scores shape: (1, n_users) -> (n_users,)
    scores = scores.view(-1)

    # 3. 상위 K명 유저 추출
    values, top_indices = torch.topk(scores, k)
    top_indices = top_indices.cpu().numpy()

    # 4. 결과 출력
    print(f"\nTop-{k} 추천 유저 리스트:")
    print("-" * 30)
    for rank, idx in enumerate(top_indices):
        # RecBole 내부 ID를 실제 User ID로 변환 (필요한 경우)
        # flowcf.yaml에서 ITEM_ID_FIELD: uid 로 설정했으므로, 
        # dataset.id2token(dataset.iid_field, idx)를 쓰면 실제 uid가 나옵니다.
        real_uid = dataset.id2token(dataset.iid_field, idx)
        score = values[rank].item()
        print(f"Rank {rank+1}: User ID {real_uid} (Score: {score:.4f})")
    print("-" * 30)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='flowcf.yaml')
    parser.add_argument('--checkpoint', type=str, required=True, help='학습된 모델 경로 (.pth)')
    parser.add_argument('--mid', type=int, required=True, help='추론할 영화 ID (Mid)')
    args = parser.parse_args()

    # 1. 설정 로드
    config = Config(model=FlowCF, config_file_list=[args.config])
    init_seed(config['seed'], config['reproducibility'])
    
    # 2. 데이터셋 정보 로드 (ID 매핑 정보를 위해 필요)
    dataset = create_dataset(config)
    
    # 3. 모델 초기화
    # (데이터셋의 메타데이터만 필요하므로 전체 로딩 없이 dataset만 넘김)
    model = FlowCF(config, dataset).to(config['device'])
    
    # 4. 체크포인트 로드
    print(f"Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # 5. 추론 실행
    inference_cold_item(model, dataset, args.mid)