# src/model.py
# v2: LayerNorm + SiLU + AdaLN (시간 조건 개선)
# 기존 FlowModel과 동일한 인터페이스 유지 → plot_efficiency.py 변경 불필요
import tensorflow as tf
import math


def timestep_embedding(timesteps, dim, max_period=10000):
    """Sinusoidal Timestep Embedding (동일)"""
    half = dim // 2
    freqs = tf.exp(-math.log(max_period) * tf.range(0, half, dtype=tf.float32) / half) * 2 * math.pi
    t = tf.squeeze(tf.cast(timesteps, tf.float32), axis=-1)
    args = t[:, None] * freqs[None]
    embedding = tf.concat([tf.cos(args), tf.sin(args)], axis=-1)
    if dim % 2:
        embedding = tf.concat([embedding, tf.zeros_like(embedding[:, :1])], axis=-1)
    return embedding


class AdaLNBlock(tf.keras.layers.Layer):
    """Adaptive Layer Normalization Block
    
    시간 임베딩이 LayerNorm의 scale(γ)과 shift(β)를 동적으로 조절.
    DiT (Diffusion Transformer) 논문에서 concat 대비 큰 성능 향상 보고.
    
    구조: LayerNorm → Dense → SiLU → Dropout
          t_emb → Dense → [scale, shift] → LayerNorm에 적용
    """
    def __init__(self, hidden_dim, time_emb_dim, dropout_rate=0.0):
        super().__init__()
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dense = tf.keras.layers.Dense(hidden_dim)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        
        # AdaLN: 시간 임베딩 → scale, shift (2 * hidden_dim)
        self.adaln_proj = tf.keras.layers.Dense(hidden_dim * 2,
                                                 kernel_initializer='zeros')
    
    def call(self, h, t_emb, training=False):
        # 1) Dense → SiLU
        h_out = self.dense(h)
        h_out = h_out * tf.sigmoid(h_out)  # SiLU = x * sigmoid(x)
        
        # 2) AdaLN: 시간 조건으로 scale/shift 생성
        adaln_params = self.adaln_proj(t_emb)  # (batch, hidden_dim * 2)
        scale, shift = tf.split(adaln_params, 2, axis=-1)  # 각 (batch, hidden_dim)
        scale = 1.0 + scale  # scale은 1 근처에서 시작
        
        # 3) LayerNorm 후 scale/shift 적용
        h_out = self.norm(h_out)
        h_out = h_out * scale + shift
        
        # 4) Dropout
        h_out = self.dropout(h_out, training=training)
        
        return h_out


class FlowModel(tf.keras.Model):
    """Flow Matching / DDPM용 MLP 모델 (v2)
    
    개선사항 (v1 대비):
    1. LayerNorm: 입력 스케일 정규화 (x_t, cond, t_emb 스케일 통일)
    2. SiLU 활성화: 현대 diffusion 모델 표준 (DDPM, DiT, Stable Diffusion)
    3. AdaLN: 시간 정보를 concat이 아닌 scale/shift로 주입 → 더 강한 조건 반영
    4. Residual Connection: hidden layer 간 잔차 연결 (학습 안정성)
    
    인터페이스: FlowModel(dims, time_emb_dim, dropout_rate)
               model(x_t, cond, t, training=False) → 기존과 동일
    """
    def __init__(self, dims, time_emb_dim, dropout_rate=0.0):
        super(FlowModel, self).__init__()
        self.time_emb_dim = time_emb_dim
        self.num_hidden = len(dims) - 1  # 마지막은 출력 차원

        # 시간 임베딩 MLP: sinusoidal → Dense → SiLU → Dense → SiLU
        self.time_mlp = [
            tf.keras.layers.Dense(time_emb_dim * 4),  # 확장
            tf.keras.layers.Dense(time_emb_dim * 4),   # 유지
        ]

        # 입력 프로젝션: concat([x_t, cond]) → hidden_dim
        # (시간 정보는 AdaLN으로 주입하므로 concat에서 제외)
        self.input_proj = tf.keras.layers.Dense(dims[0])
        self.input_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Hidden blocks: AdaLN 방식
        self.blocks = []
        for i in range(self.num_hidden):
            self.blocks.append(
                AdaLNBlock(dims[i], time_emb_dim * 4, dropout_rate)
            )

        # 출력 레이어
        self.output_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.output_proj = tf.keras.layers.Dense(dims[-1])  # num_users

    def call(self, x_t, cond, t, training=False):
        # Mixed Precision 호환
        compute_dtype = self.compute_dtype
        x_t = tf.cast(x_t, compute_dtype)
        cond = tf.cast(cond, compute_dtype)
        t = tf.cast(t, tf.float32)

        # ── 시간 임베딩 ──
        t_emb = timestep_embedding(t, self.time_emb_dim)
        t_emb = tf.cast(t_emb, compute_dtype)
        for layer in self.time_mlp:
            t_emb = layer(t_emb)
            t_emb = t_emb * tf.sigmoid(t_emb)  # SiLU

        # ── 입력 프로젝션 (시간 제외) ──
        h = tf.concat([x_t, cond], axis=-1)
        h = self.input_proj(h)
        h = h * tf.sigmoid(h)  # SiLU
        h = self.input_norm(h)

        # ── Hidden Blocks (AdaLN + Residual) ──
        for block in self.blocks:
            h_res = h
            h = block(h, t_emb, training=training)
            h = h + h_res  # Residual connection

        # ── 출력 ──
        h = self.output_norm(h)
        h = self.output_proj(h)

        return tf.cast(h, tf.float32)