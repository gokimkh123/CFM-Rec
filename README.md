ColdStart-FlowCF-TF/
├── data/                 # 원본 데이터 및 전처리된 데이터
│   ├── ML1M/             # 무비렌즈 1M 데이터셋
│   └── side_info.npy     # 아이템 태그 임베딩
├── src/                  # 소스 코드 모음
│   ├── __init__.py
│   ├── data_loader.py    # tf.data를 활용한 데이터 파이프라인
│   ├── model.py          # FlowModel (MLP + Time Emb) 정의
│   ├── flow_logic.py     # FM 수식 (Gaussian/Bernoulli) 정의
│   └── metrics.py        # Recall, NDCG 등 평가 지표 구현
├── configs/
│   └── base_config.yaml  # 하이퍼파라미터 설정
├── train.py              # GradientTape 기반의 진짜 학습 루프
├── evaluate.py           # Cold-start 전용 평가 스크립트
└── inference.py          # 특정 아이템 기반 유저 추천 추론
