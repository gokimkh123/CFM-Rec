# src/model.py (수정본)
import tensorflow as tf
import math

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = tf.exp(-math.log(max_period) * tf.range(0, half, dtype=tf.float32) / half) * 2 * math.pi
    t = tf.squeeze(tf.cast(timesteps, tf.float32), axis=-1) 
    args = t[:, None] * freqs[None]
    embedding = tf.concat([tf.cos(args), tf.sin(args)], axis=-1)
    if dim % 2:
        embedding = tf.concat([embedding, tf.zeros_like(embedding[:, :1])], axis=-1)
    return embedding

class FlowModel(tf.keras.Model):
    def __init__(self, dims, time_emb_dim, dropout_rate=0.0):
        super(FlowModel, self).__init__()
        self.time_emb_dim = time_emb_dim

        # 시간 임베딩 처리
        self.time_dense = tf.keras.layers.Dense(time_emb_dim)
        self.time_act = tf.keras.layers.Activation('swish') # [변경] LeakyReLU -> Swish

        self.mlp_layers = []
        
        # dims 리스트에서 '마지막 차원(Output)'을 분리해냅니다.
        # 예: dims = [300, 300, 6039] 라면
        # hidden_dims = [300, 300], output_dim = 6039
        hidden_dims = dims[:-1]
        output_dim = dims[-1]

        # 1. Hidden Layers (압축 및 연산 구간)
        for dim in hidden_dims:
            self.mlp_layers.append(tf.keras.layers.Dense(dim))
            self.mlp_layers.append(tf.keras.layers.LayerNormalization()) # [정규화 O]
            self.mlp_layers.append(tf.keras.layers.Activation('swish'))  # [DiffRec 스타일]
            if dropout_rate > 0:
                self.mlp_layers.append(tf.keras.layers.Dropout(dropout_rate))

        # 2. Output Layer (최종 출력 구간)
        # 여기는 정규화를 하면 안 됩니다! Raw 값을 뱉어야 합니다.
        self.output_layer = tf.keras.layers.Dense(output_dim)
        
        # (선택) 만약 DiffRec처럼 값을 -1~1로 가두고 싶다면 주석 해제
        # self.final_act = tf.keras.layers.Activation('tanh') 

    def call(self, x_t, cond, t, training=False):
            # 1. Time & Input Embedding (동일)
            t_emb = timestep_embedding(t, self.time_emb_dim)
            t_emb = self.time_dense(t_emb)
            t_emb = self.time_act(t_emb)
            
            h = tf.concat([x_t, cond, t_emb], axis=-1)

            # 2. Hidden Layers (Swish로 부드럽게 추론)
            for layer in self.mlp_layers:
                if isinstance(layer, tf.keras.layers.Dropout):
                    h = layer(h, training=training)
                else:
                    h = layer(h)
            
            # 3. Output Layer (차원 복원)
            out = self.output_layer(h)
            
            out = tf.tanh(out) 
            
            return out