# src/flow_logic.py
import tensorflow as tf

class BernoulliFlow:
    def __init__(self, user_activity, prior_type='popularity'):
            self.user_activity = tf.convert_to_tensor(user_activity, dtype=tf.float32)
            self.prior_type = prior_type # [추가]
    def get_prior_sample(self, batch_size):
            if self.prior_type == 'noise':
                # 아무 정보 없는 상태 (모든 유저가 50% 확률로 0 또는 1)
                probs = tf.fill([batch_size, tf.shape(self.user_activity)[0]], 0.5)
            else:
                # 기존 인기 기반 분포
                probs = tf.tile(tf.expand_dims(self.user_activity, 0), [batch_size, 1])
                
            return tf.cast(tf.random.uniform(tf.shape(probs)) < probs, tf.float32)

    def inference_step(self, x_t, pred, t, dt):
        """Bernoulli 자가 교정 추론 스텝"""
        # t는 스칼라값 혹은 (1,1) 형태
        v_t = (pred - x_t) / (1.0 - tf.cast(t, tf.float32) + 1e-5)
        return x_t + v_t * dt