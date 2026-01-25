import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils.enum_type import InputType
import math
import os

class FlowModel(nn.Module):
    def __init__(self, dims, time_emb_size, time_type="cat", dropout_prob=0.0, act_func="gelu", norm=False):
        super(FlowModel, self).__init__()
        self.dims = dims
        self.time_type = time_type
        self.time_emb_dim = time_emb_size
        self.norm = norm
        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)
        
        # 활성화 함수 선택
        if act_func.lower() == 'leakyrelu':
            activation = nn.LeakyReLU(negative_slope=0.2)
        else:
            activation = nn.GELU()

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(activation)
                layers.append(nn.Dropout(p=dropout_prob)) # <--- 설정값 사용하도록 변경
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, t):
        time_emb = timestep_embedding_pi(t, self.time_emb_dim).to(x.device)
        time_emb = self.emb_layer(time_emb)
        if self.time_type == "cat":
            x = torch.cat([x, time_emb], dim=-1)
        out = self.mlp(x)
        return out

class FlowCF(GeneralRecommender):
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(FlowCF, self).__init__(config, dataset)

        # 1. 설정 로드
        if 'prior_type' in config:
            self.prior_type = config['prior_type'].lower()
        else:
            self.prior_type = 'gaussian'

        if 'act_func' in config:
            self.act_func = config['act_func'].lower()
        else:
            self.act_func = 'gelu'
        
        print(f"\n[Model Config] Prior: {self.prior_type.upper()} | Activation: {self.act_func.upper()}\n")

        # 2. Side Info 로드 및 정렬
        npy_path = os.path.join("dataset", "ML1M", "mv-tag-emb.npy")
        if not os.path.exists(npy_path):
            npy_path = "/app/dataset/ML1M/mv-tag-emb.npy"
        
        raw_emb = np.load(npy_path)
        self.raw_side_emb = torch.FloatTensor(raw_emb).to(self.device)
        self.side_dim = self.raw_side_emb.shape[1]

        # Re-alignment
        n_movies = self.n_users 
        aligned_emb = torch.zeros((n_movies, self.side_dim)).to(self.device)
        for internal_id in range(1, n_movies):
            try:
                raw_token = dataset.id2token(dataset.uid_field, internal_id)
                raw_id = int(raw_token)
                if 0 <= raw_id < self.raw_side_emb.shape[0]:
                    aligned_emb[internal_id] = self.raw_side_emb[raw_id]
            except:
                pass
        self.side_emb = aligned_emb

        # [순서 수정됨] 3. History Matrix 생성 (이게 먼저 있어야 계산 가능)
        inter_matrix = dataset.inter_matrix(form='csr').astype('float32')
        self.history_matrix = torch.FloatTensor(inter_matrix.toarray()).to(self.device)

        # 4. Bernoulli Prior를 위한 유저 활동성 계산 (History Matrix 생성 후 호출)
        if self.prior_type == 'bernoulli':
            self.user_activity = self._get_user_activity(dataset)

        # 5. 모델 구축
        self.target_dim = self.n_items 
        self.input_dim = self.target_dim + self.side_dim
        self.n_steps = config["n_steps"]
        self.time_emb_size = config["time_embedding_size"]
        
        self.dims_mlp = [self.input_dim + self.time_emb_size] + config["dims_mlp"] + [self.target_dim]

        self.net = FlowModel(
            dims=self.dims_mlp,
            time_emb_size=self.time_emb_size,
            dropout_prob=config['dropout'],
            time_type="cat",
            act_func=self.act_func,
            norm=False
        ).to(self.device)

    def _get_user_activity(self, dataset):
        """각 유저가 영화를 좋아할 평균 확률 (Prior) 계산"""
        activity = self.history_matrix.mean(dim=0)
        return torch.clamp(activity, 1e-6, 1.0 - 1e-6)

    def calculate_loss(self, interaction):
        movie_ids = interaction[self.USER_ID]
        x_1 = self.history_matrix[movie_ids]
        cond = self.side_emb[movie_ids]
        batch_size = x_1.size(0)

        t = torch.rand(batch_size).to(self.device)
        t_expand = t.view(-1, 1)

        if self.prior_type == 'bernoulli':
            # Bernoulli Path
            prior_probs = self.user_activity.expand(batch_size, -1)
            x_0 = torch.bernoulli(prior_probs).to(self.device)
            mask = torch.rand_like(x_1) <= t_expand
            x_t = torch.where(mask, x_1, x_0)
            target = x_1
        else:
            # Gaussian Path
            x_0 = torch.randn_like(x_1).to(self.device)
            x_t = (1 - t_expand) * x_0 + t_expand * x_1
            target = x_1 - x_0

        net_input = torch.cat([x_t, cond], dim=1)
        pred = self.net(net_input, t)

        loss = F.mse_loss(pred, target)
        return loss

    def full_sort_predict(self, interaction):
        movie_ids = interaction[self.USER_ID]
        cond = self.side_emb[movie_ids]
        batch_size = movie_ids.size(0)

        # Init Noise
        if self.prior_type == 'bernoulli':
            prior_probs = self.user_activity.expand(batch_size, -1)
            x_t = torch.bernoulli(prior_probs).to(self.device)
        else:
            x_t = torch.randn(batch_size, self.target_dim).to(self.device)

        steps = self.n_steps
        dt = 1.0 / steps

        # ODE Solver
        for i in range(steps):
            t_scalar = i / steps
            t_tensor = torch.full((batch_size,), t_scalar).to(self.device)
            
            net_input = torch.cat([x_t, cond], dim=1)
            pred = self.net(net_input, t_tensor)
            
            if self.prior_type == 'bernoulli':
                if t_scalar >= 1.0: break
                v_t = (pred - x_t) / (1.0 - t_scalar + 1e-5)
                x_t = x_t + v_t * dt
            else:
                x_t = x_t + pred * dt
        
        return x_t

    def predict(self, interaction):
        item = interaction[self.ITEM_ID]
        x_t = self.full_sort_predict(interaction)
        scores = x_t.gather(1, item.unsqueeze(1)).squeeze(1)
        return scores

    def predict_cold_item(self, item_id_or_emb, num_samples=10):
        if isinstance(item_id_or_emb, int):
            if item_id_or_emb < self.side_emb.shape[0]:
                 cond = self.side_emb[item_id_or_emb].unsqueeze(0)
            else:
                 cond = torch.zeros(1, self.side_dim).to(self.device)
        else:
            cond = torch.FloatTensor(item_id_or_emb).to(self.device)
            if cond.dim() == 1:
                cond = cond.unsqueeze(0)

        cond_expanded = cond.repeat(num_samples, 1)
        batch_size = num_samples
        
        if self.prior_type == 'bernoulli':
            prior_probs = self.user_activity.expand(batch_size, -1)
            x_t = torch.bernoulli(prior_probs).to(self.device)
        else:
            x_t = torch.randn(batch_size, self.target_dim).to(self.device)

        steps = self.n_steps
        dt = 1.0 / steps
        
        with torch.no_grad():
            for i in range(steps):
                t_scalar = i / steps
                t_tensor = torch.full((batch_size,), t_scalar).to(self.device)
                
                net_input = torch.cat([x_t, cond_expanded], dim=1)
                pred = self.net(net_input, t_tensor)
                
                if self.prior_type == 'bernoulli':
                    if t_scalar >= 1.0: break
                    v_t = (pred - x_t) / (1.0 - t_scalar + 1e-5)
                    x_t = x_t + v_t * dt
                else:
                    x_t = x_t + pred * dt

        return x_t.mean(dim=0, keepdim=True)

def timestep_embedding_pi(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(timesteps.device) * 2 * math.pi
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding