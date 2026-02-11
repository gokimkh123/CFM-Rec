import pandas as pd
import numpy as np
import tensorflow as tf
import os

class ColdStartDataLoader:
    def __init__(self, config):
        """
        config: 전체 설정 딕셔너리 (YAML 로드 결과)
        """
        # 1. 사용할 데이터셋 이름 가져오기
        self.dataset_name = config.get('dataset')
        if not self.dataset_name:
            raise ValueError("❌ [Error] config.yaml에 'dataset' 항목이 없습니다.")

        # 2. 해당 데이터셋의 설정 블록 가져오기 (없으면 에러 발생!)
        ds_config = config.get(self.dataset_name)
        if not ds_config:
            raise ValueError(f"❌ [Error] config.yaml에 '{self.dataset_name}' 설정 블록이 없습니다.")
        
        # 3. 설정 적용 (기본값 제거 -> YAML에 경로 누락시 에러 발생 유도)
        #    이렇게 해야 경로가 잘못되었을 때 바로 알 수 있습니다.
        try:
            self.data_path = ds_config['data_path']
            self.train_file = ds_config.get('train_file', 'train.csv') # 파일명은 기본값 있어도 됨
            self.vali_file = ds_config.get('vali_file', 'vali.csv')
            self.test_file = ds_config.get('test_file', 'test.csv')
            
            # 여기가 핵심: Books는 config에서 side_info_path를 꼭 지정했으므로 가져와야 함
            self.side_info_path = ds_config['side_info_path'] 
            
            self.separator = ds_config.get('separator', ',') # 구분자
            self.entity_field = ds_config.get('entity_field', 'iid')
            self.target_field = ds_config.get('target_field', 'uid')

        except KeyError as e:
            raise KeyError(f"❌ [Error] '{self.dataset_name}' 설정에 필수 항목 {e} 가 누락되었습니다.")
        
        self.head_drop_ratio = config.get('head_drop_ratio', 0.0)
        self.batch_size = config.get('batch_size', 1024)

        print(f">>> [DataLoader] Initialized for dataset: {self.dataset_name}")
        print(f"    - Path: {self.data_path}")
        print(f"    - Side Info: {self.side_info_path}")
        print(f"    - Separator: '{self.separator}'")

        self.vali_item_ids = []
        self.test_item_ids = []

    def _load_inter(self, file_name):
        path = os.path.join(self.data_path, file_name)
        print(f"    Loading: {path}")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        # 구분자 자동 감지 로직
        sep = self.separator
        if sep is None:
            if file_name.endswith('.tsv'):
                sep = '\t'
            else:
                sep = ','

        try:
            df = pd.read_csv(path, sep=sep)
        except Exception as e:
            print(f"    Error reading {file_name} with sep='{sep}': {e}")
            alt_sep = ',' if sep == '\t' else '\t'
            print(f"    Retrying with sep='{alt_sep}'...")
            df = pd.read_csv(path, sep=alt_sep)

        # 컬럼명 정규화
        rename_map = {}
        for col in df.columns:
            c_lower = col.lower()
            if self.target_field in c_lower or 'user' in c_lower or 'uid' in c_lower:
                rename_map[col] = self.target_field
            elif self.entity_field in c_lower or 'item' in c_lower or 'mid' in c_lower or 'iid' in c_lower:
                rename_map[col] = self.entity_field
        
        if rename_map:
            df = df.rename(columns=rename_map)

        # 필수 컬럼 확인 및 대체
        if self.entity_field not in df.columns or self.target_field not in df.columns:
            print(f"    Warning: Columns not found via name. Using 1st(User) and 2nd(Item) columns.")
            # 데이터프레임 복사본 생성 방지를 위해 바로 할당
            cols = list(df.columns)
            df.columns = [self.target_field, self.entity_field] + cols[2:]

        # ID를 문자열로 변환
        df[self.entity_field] = df[self.entity_field].astype(str)
        df[self.target_field] = df[self.target_field].astype(str)
        
        return df

    def _build_matrix(self, df):
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
        """
        데이터를 로드하고 매트릭스를 생성합니다.
        Returns:
            num_entities (int): 아이템 수
            num_targets (int): 유저 수
        """
        # 1. 데이터 로드
        train_df = self._load_inter(self.train_file)
        vali_df = self._load_inter(self.vali_file)
        test_df = self._load_inter(self.test_file)
        
        # 2. ID 매핑
        full_df = pd.concat([train_df, vali_df, test_df])
        self.entity_tokens = sorted(full_df[self.entity_field].unique())
        self.target_tokens = sorted(full_df[self.target_field].unique())
        
        self.entity2id = {t: i for i, t in enumerate(self.entity_tokens)}
        self.target2id = {t: i for i, t in enumerate(self.target_tokens)}
        
        self.num_entities = len(self.entity_tokens)
        self.num_targets = len(self.target_tokens)
        
        print(f">>> [Stats] Total Items: {self.num_entities}, Total Users: {self.num_targets}")

        # 3. Cold Item 식별
        self.vali_item_ids = sorted(list(set(
            [self.entity2id[t] for t in vali_df[self.entity_field].unique() if t in self.entity2id]
        )))
        self.test_item_ids = sorted(list(set(
            [self.entity2id[t] for t in test_df[self.entity_field].unique() if t in self.entity2id]
        )))

        # 4. Matrix 생성
        print(">>> Building Interaction Matrices...")
        self.train_matrix = self._build_matrix(train_df)
        self.vali_matrix = self._build_matrix(vali_df)
        self.test_matrix = self._build_matrix(test_df)

        # 5. Side Info 로드
        print(f">>> Loading Side Info from: {self.side_info_path}")
        if os.path.exists(self.side_info_path):
            raw_side_emb = np.load(self.side_info_path)
            self.side_dim = raw_side_emb.shape[1]
            self.side_emb = np.zeros((self.num_entities, self.side_dim), dtype=np.float32)
            
            hit_count = 0
            for token, eid in self.entity2id.items():
                try:
                    # 토큰이 숫자형 ID인 경우
                    raw_idx = int(token)
                    if raw_idx < len(raw_side_emb):
                        self.side_emb[eid] = raw_side_emb[raw_idx]
                        hit_count += 1
                except ValueError:
                    pass
            print(f">>> Side Info Loaded. Mapped {hit_count}/{self.num_entities} items.")
        else:
            print(f"⚠️ Warning: Side info file not found. Initializing with zeros.")
            self.side_dim = 128
            self.side_emb = np.zeros((self.num_entities, self.side_dim), dtype=np.float32)

        # 6. User Activity (Prior용)
        user_activity = self.train_matrix.mean(axis=0)
        self.user_activity = np.clip(user_activity, 1e-6, 1.0 - 1e-6)

        # train.py 호환성을 위해 개수 반환
        return self.num_entities, self.num_targets

    def get_dataset(self, mode='train'):
        if mode == 'train':
            entity_ids = np.arange(self.num_entities)
            target_matrix = self.train_matrix
            shuffle = True
        elif mode == 'vali':
            entity_ids = np.array(self.vali_item_ids)
            target_matrix = self.vali_matrix
            shuffle = False
        elif mode == 'test':
            entity_ids = np.array(self.test_item_ids)
            target_matrix = self.test_matrix
            shuffle = False
        else:
            raise ValueError(f"Unknown mode: {mode}")

        def generator():
            for eid in entity_ids:
                yield target_matrix[eid], self.side_emb[eid]

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(self.num_targets,), dtype=tf.float32),
                tf.TensorSpec(shape=(self.side_dim,), dtype=tf.float32)
            )
        )

        if shuffle:
            dataset = dataset.shuffle(self.num_entities).batch(self.batch_size)
        else:
            dataset = dataset.batch(self.batch_size)

        return dataset.prefetch(tf.data.AUTOTUNE)