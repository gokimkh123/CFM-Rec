import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils.enum_type import InputType
import math
import os

class DiffusionNet(nn.Module):
    def __init__(self, dims, time_emb_size, act_func="gelu"):
        super(DiffusionNet, self).__init__()
        self.dims = dims
        self.time_emb_dim = time_emb_size
        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)
        
        # 활성화 함수 선택 로직
        if act_func.lower() == 'leakyrelu':
            activation = nn.LeakyReLU(negative_slope=0.2)
        else:
            activation = nn.GELU()

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(activation)
                layers.append(nn.Dropout(p=0.2)) 
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, t):
        time_emb = timestep_embedding_pi(t, self.time_emb_dim).to(x.device)
        time_emb = self.emb_layer(time_emb)
        
        x = torch.cat([x, time_emb], dim=-1)
        out = self.mlp(x)
        return out

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
class DiffCF(GeneralRecommender):
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(DiffCF, self).__init__(config, dataset)

        # =========================================================
        # 1. 설정값 로드 (이 부분이 빠져서 에러가 났습니다)
        # =========================================================
        self.n_steps = config["n_steps"]
        
        # config에서 값을 읽어오되, 없으면 기본값(0.0001, 0.02) 사용
        if 'beta_start' in config:
            self.beta_start = config['beta_start']
        else:
            self.beta_start = 0.0001

        if 'beta_end' in config:
            self.beta_end = config['beta_end']
        else:
            self.beta_end = 0.02

        if 'act_func' in config:
            self.act_func = config['act_func']
        else:
            self.act_func = 'gelu'

        # =========================================================
        # 2. Beta Schedule 설정 (가우시안 확산 스케줄)
        # =========================================================
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.n_steps).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        
        # [x_0 예측을 위한 Posterior 계산용 계수들]
        # alphas_cumprod_prev: [1.0, alpha_1, alpha_2, ...] (한 칸씩 밀기)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Posterior Mean 계산 계수
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
        
        # Posterior Variance (log로 저장하여 안정성 확보)
        posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(posterior_variance, min=1e-20)
        )

        # =========================================================
        # 3. Side Info 로드 (FlowCF와 동일)
        # =========================================================
        npy_path = os.path.join("dataset", "ML1M", "mv-tag-emb.npy")
        if not os.path.exists(npy_path):
            npy_path = "/app/dataset/ML1M/mv-tag-emb.npy"
        
        raw_emb = np.load(npy_path)
        self.raw_side_emb = torch.FloatTensor(raw_emb).to(self.device)
        self.side_dim = self.raw_side_emb.shape[1]

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

        inter_matrix = dataset.inter_matrix(form='csr').astype('float32')
        self.history_matrix = torch.FloatTensor(inter_matrix.toarray()).to(self.device)
        
        # =========================================================
        # 4. 모델 네트워크 구축
        # =========================================================
        self.target_dim = self.n_items 
        self.input_dim = self.target_dim + self.side_dim 
        self.time_emb_size = config["time_embedding_size"]
        
        self.dims_mlp = [self.input_dim + self.time_emb_size] + config["dims_mlp"] + [self.target_dim]

        self.net = DiffusionNet(
            dims=self.dims_mlp,
            time_emb_size=self.time_emb_size,
            act_func=self.act_func
        ).to(self.device)
        
    # [수정 2] Loss 계산: 정답(Target)을 노이즈가 아닌 '원본(x_0)'으로 변경
    def calculate_loss(self, interaction):
        movie_ids = interaction[self.USER_ID]
        x_0 = self.history_matrix[movie_ids]  # 원본 데이터
        cond = self.side_emb[movie_ids]     
        batch_size = x_0.size(0)

        t = torch.randint(0, self.n_steps, (batch_size,), device=self.device).long()
        noise = torch.randn_like(x_0).to(self.device)

        # Forward Process (x_0 -> x_t)
        # x_t = sqrt(alpha_bar) * x_0 + sqrt(1-alpha_bar) * noise
        sqrt_alpha_cumprod = torch.sqrt(self.alphas_cumprod[t]).view(-1, 1)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alphas_cumprod[t]).view(-1, 1)
        
        x_t = sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise

        # 모델 예측
        net_input = torch.cat([x_t, cond], dim=1)
        predicted_x0 = self.net(net_input, t) # 모델이 x_0를 직접 예측하도록 함

        # Loss: 예측한 x_0와 실제 x_0 비교
        loss = F.mse_loss(predicted_x0, x_0)
        return loss

    # [수정 3] 추론 과정: 예측된 x_0를 이용해 x_{t-1} 계산 (DiffRec 논문 Eq.10 방식)
    @torch.no_grad()
    def p_sample_loop(self, interaction):
        movie_ids = interaction[self.USER_ID]
        cond = self.side_emb[movie_ids]
        batch_size = movie_ids.size(0)

        # 랜덤 노이즈에서 시작
        x_t = torch.randn(batch_size, self.target_dim).to(self.device)

        for t in reversed(range(self.n_steps)):
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            
            # 1. 모델이 x_0를 예측 ("이 노이즈 낀 데이터의 원본은 이거야!")
            net_input = torch.cat([x_t, cond], dim=1)
            predicted_x0 = self.net(net_input, t_tensor)
            
            # (옵션) 예측값 클리핑: 데이터가 0~1 사이라면 클리핑하면 성능이 좋아질 수 있음
            # predicted_x0 = torch.clamp(predicted_x0, -1.0, 1.0) 

            # 2. Posterior Mean 계산 (논문 Eq.7, Eq.10)
            # mu_t = coef1 * pred_x0 + coef2 * x_t
            posterior_mean = (
                self.posterior_mean_coef1[t] * predicted_x0 +
                self.posterior_mean_coef2[t] * x_t
            )

            # 3. 다음 단계 샘플링 (x_{t-1})
            if t > 0:
                noise = torch.randn_like(x_t)
                # sigma_t = sqrt(posterior_variance)
                posterior_log_variance = self.posterior_log_variance_clipped[t]
                x_t = posterior_mean + torch.exp(0.5 * posterior_log_variance) * noise
            else:
                x_t = posterior_mean # 마지막 단계는 노이즈 없이 평균값 사용

        return x_t