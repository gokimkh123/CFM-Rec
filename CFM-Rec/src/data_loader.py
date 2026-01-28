import pandas as pd
import numpy as np
import tensorflow as tf
import os

class ColdStartDataLoader:
    def __init__(self, config):
        self.config = config
        self.data_path = config['data_path']
        self.side_info_path = config['side_info_path']
        self.entity_field = config['entity_field']
        self.target_field = config['target_field']
        
        self.head_drop_ratio = config.get('head_drop_ratio', 0.0)
        self.batch_size = config.get('batch_size', 2048)

        # [수정 1] 필터링된 아이템 ID를 저장할 리스트 초기화
        self.vali_item_ids = []
        self.test_item_ids = []

    def _load_inter(self, file_name):
        path = os.path.join(self.data_path, file_name)
        # ID 타입 불일치 방지를 위해 모든 ID를 문자열(str)로 읽음
        df = pd.read_csv(path, sep='\t', dtype={
            f"{self.config['entity_field']}:token": str, 
            f"{self.config['target_field']}:token": str
        })
        df.columns = [col.split(':')[0] for col in df.columns]
        return df

    def _build_matrix(self, df):
        """특정 데이터프레임을 Interaction Matrix로 변환"""
        mat = np.zeros((self.num_entities, self.num_targets), dtype=np.float32)
        for _, row in df.iterrows():
            e_token = row[self.entity_field]
            t_token = row[self.target_field]
            if e_token in self.entity2id and t_token in self.target2id:
                eid = self.entity2id[e_token]
                tid = self.target2id[t_token]
                mat[eid, tid] = 1.0
        return mat

    def build(self):
        # 1. 데이터 로드
        train_df = self._load_inter(self.config['train_file'])
        vali_df = self._load_inter(self.config['vali_file'])
        test_df = self._load_inter(self.config['test_file'])
        
        # 2. ID 매핑 (전체 데이터셋 기준)
        full_df = pd.concat([train_df, vali_df, test_df])
        self.entity_tokens = sorted(full_df[self.entity_field].unique())
        self.target_tokens = sorted(full_df[self.target_field].unique())
        
        self.entity2id = {t: i for i, t in enumerate(self.entity_tokens)}
        self.target2id = {t: i for i, t in enumerate(self.target_tokens)}
        
        self.num_entities = len(self.entity_tokens)
        self.num_targets = len(self.target_tokens)

        # [수정 2] Vali와 Test에 실제로 등장하는 아이템 ID만 추출 (Cold Item)
        # evaluate.py의 valid_test_items 로직과 동일하게 만듭니다.
        self.vali_item_ids = sorted(list(set(
            [self.entity2id[t] for t in vali_df[self.entity_field].unique() if t in self.entity2id]
        )))
        self.test_item_ids = sorted(list(set(
            [self.entity2id[t] for t in test_df[self.entity_field].unique() if t in self.entity2id]
        )))

        print(f">>> [Filter] Vali Items: {len(self.vali_item_ids)} / Test Items: {len(self.test_item_ids)}")
        
        # 3. 데이터셋별 매트릭스 분리 생성
        print(">>> 데이터 매트릭스 변환 중...")
        self.train_matrix = self._build_matrix(train_df)
        self.vali_matrix = self._build_matrix(vali_df)
        self.test_matrix = self._build_matrix(test_df) # Test Matrix도 미리 생성

        # 4. Masking (Train Matrix에만 적용)
        if self.head_drop_ratio > 0:
            popularity = self.train_matrix.sum(axis=1)
            num_to_drop = int(self.num_entities * self.head_drop_ratio)
            if num_to_drop > 0:
                top_indices = np.argsort(popularity)[-num_to_drop:]
                self.train_matrix[top_indices] = 0.0
                print(f">>> [Masking] Train: 상위 {self.head_drop_ratio*100}% 아이템({num_to_drop}개) 마스킹.")

        # 5. Side Info 로드
        raw_side_emb = np.load(self.side_info_path)
        self.side_dim = raw_side_emb.shape[1]
        self.side_emb = np.zeros((self.num_entities, self.side_dim), dtype=np.float32)
        
        for token, eid in self.entity2id.items():
            try:
                raw_idx = int(token)
                if raw_idx < len(raw_side_emb):
                    self.side_emb[eid] = raw_side_emb[raw_idx]
            except ValueError:
                pass

        # 6. User Activity 계산 (Train 기준)
        user_activity = self.train_matrix.mean(axis=0)
        self.user_activity = np.clip(user_activity, 1e-6, 1.0 - 1e-6)
        
        print(f">>> 데이터 빌드 완료: Items={self.num_entities}, Users={self.num_targets}")
        return self.num_entities, self.num_targets

    def get_dataset(self, mode='train'):
        """학습(train) 또는 검증(vali) 데이터셋 반환"""
        
        # [수정 3] 모드에 따라 평가할 아이템(Entity) 범위를 제한
        if mode == 'train':
            target_matrix = self.train_matrix
            entity_ids = np.arange(self.num_entities) # Train은 전체 셔플
        elif mode == 'vali':
            target_matrix = self.vali_matrix
            entity_ids = np.array(self.vali_item_ids) # Vali 파일에 있는 아이템만!
        elif mode == 'test':
            target_matrix = self.test_matrix
            entity_ids = np.array(self.test_item_ids) # Test 파일에 있는 아이템만!
        else:
            # 기본값 (혹시 모를 오류 방지)
            target_matrix = self.train_matrix
            entity_ids = np.arange(self.num_entities)
        
        def generator():
            for eid in entity_ids:
                # (정답 벡터, 조건 벡터)
                yield target_matrix[eid], self.side_emb[eid]

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(self.num_targets,), dtype=tf.float32),
                tf.TensorSpec(shape=(self.side_dim,), dtype=tf.float32)
            )
        )
        
        if mode == 'train':
            dataset = dataset.shuffle(self.num_entities).batch(self.batch_size)
        else:
            # 평가 시에는 순서대로 (셔플 X)
            dataset = dataset.batch(self.batch_size)
            
        return dataset.prefetch(tf.data.AUTOTUNE)