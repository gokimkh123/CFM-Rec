# src/flow_logic.py
import tensorflow as tf

class BernoulliFlow:
    def __init__(self, user_activity, prior_type='popularity'):
            self.user_activity = tf.convert_to_tensor(user_activity, dtype=tf.float32)
            self.prior_type = prior_type 
    def get_prior_sample(self, batch_size):
            if self.prior_type == 'noise':
                # 아무 정보 없는 상태 (모든 유저가 50% 확률로 0 또는 1)
                probs = tf.fill([batch_size, tf.shape(self.user_activity)[0]], 0.5)
            else:
                # 기존 인기 기반 분포
                probs = tf.tile(tf.expand_dims(self.user_activity, 0), [batch_size, 1])
                
            return tf.cast(tf.random.uniform(tf.shape(probs)) < probs, tf.float32)
    def inference_step(self, x_t, pred, t, dt):
            t_float = tf.cast(t, tf.float32)
            
            # 마지막 스텝이면 모델 예측값으로 바로 점프
            if (1.0 - t_float) <= (dt + 1e-5):
                return tf.cast(pred, tf.float32)  # FP16 안전: mixed precision 시에도 float32 보장

            denom = (1.0 - t_float + 1e-5)
            v_t = (pred - x_t) / denom
            return x_t + v_t * dt