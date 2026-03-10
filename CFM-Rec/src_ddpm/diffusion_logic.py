# src_ddpm/diffusion_logic.py
import tensorflow as tf
import numpy as np

class GaussianDiffusion:
    def __init__(self, user_activity, steps=300, beta_start=0.000333, beta_end=0.0666, prior_type='popularity'):
        self.steps = steps
        # [핵심] 인기 점수를 사전 분포의 평균으로 사용하기 위해 저장
        self.user_activity = tf.convert_to_tensor(user_activity, dtype=tf.float32)
        self.prior_type = prior_type


        
        # Beta Schedule (Linear)
        self.betas = np.linspace(beta_start, beta_end, steps, dtype=np.float32)


        '''
        # [안전장치 1] 확률은 1.0을 넘을 수 없으므로 자름 (0.06 수준이라 안 걸리겠지만 안전하게)
        self.betas = np.clip(self.betas, a_min=0.0001, a_max=0.999)

        # [안전장치 2: 핵심] 마지막 스텝(x_300)은 무조건 완전 노이즈(정보량 0)가 되도록 강제
        # 이 코드가 있어야 "Linear 스케줄의 누수 문제"가 100% 차단됩니다.
        self.betas[-1] = 1.0
        '''


        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

        self.betas = tf.constant(self.betas, dtype=tf.float32)
        self.alphas_cumprod = tf.constant(self.alphas_cumprod, dtype=tf.float32)
        self.alphas_cumprod_prev = tf.constant(self.alphas_cumprod_prev, dtype=tf.float32)
        self.sqrt_alphas_cumprod = tf.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = tf.sqrt(1.0 - self.alphas_cumprod)
        
        # Reverse Process 계수 설정
        self.posterior_mean_coef1 = (
            tf.sqrt(self.alphas_cumprod_prev) * self.betas / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            tf.sqrt(self.alphas) * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
    '''
    def __init__(self, user_activity, steps=100, beta_start=1e-4, beta_end=0.02, prior_type='popularity'):
            self.steps = steps
            # [핵심] 인기 점수를 사전 분포의 평균으로 사용하기 위해 저장
            self.user_activity = tf.convert_to_tensor(user_activity, dtype=tf.float32)
            self.prior_type = prior_type

            # ============================================================
            # [변경] Beta Schedule: Linear -> Cosine
            # Cosine Schedule은 beta_start, beta_end를 직접 쓰지 않고
            # steps와 offset(s)을 이용해 계산합니다.
            # ============================================================
            s = 0.008
            
            # 1. 0부터 steps까지 (steps+1개) 구간 생성
            t = np.linspace(0, steps, steps + 1)
            
            # 2. Cosine 함수 적용 (f(t))
            # f(t) = cos^2( (t/T + s) / (1+s) * pi/2 )
            alphas_cumprod_temp = np.cos(((t / steps) + s) / (1 + s) * np.pi / 2) ** 2
            
            # 3. 첫 번째 값으로 나누어 정규화 (alpha_bar_0 = 1이 되도록)
            alphas_cumprod_temp = alphas_cumprod_temp / alphas_cumprod_temp[0]
            
            # 4. Beta 계산: beta_t = 1 - (alpha_bar_t / alpha_bar_{t-1})
            betas = 1 - (alphas_cumprod_temp[1:] / alphas_cumprod_temp[:-1])
            
            # 5. Beta 값 클리핑 (너무 커지지 않도록 최대 0.999로 제한)
            self.betas = np.clip(betas, a_min=0, a_max=0.999).astype(np.float32)
            self.betas[-1] = 1.0
            # ------------------------------------------------------------
            # 이하 로직은 동일 (self.betas를 사용하여 나머지 계산)
            # ------------------------------------------------------------
            self.alphas = 1.0 - self.betas
            self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
            self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

            self.betas = tf.constant(self.betas, dtype=tf.float32)
            self.alphas_cumprod = tf.constant(self.alphas_cumprod, dtype=tf.float32)
            self.alphas_cumprod_prev = tf.constant(self.alphas_cumprod_prev, dtype=tf.float32)
            self.sqrt_alphas_cumprod = tf.sqrt(self.alphas_cumprod)
            self.sqrt_one_minus_alphas_cumprod = tf.sqrt(1.0 - self.alphas_cumprod)
            
            # Reverse Process 계수 설정
            self.posterior_mean_coef1 = (
                tf.sqrt(self.alphas_cumprod_prev) * self.betas / (1.0 - self.alphas_cumprod)
            )
            self.posterior_mean_coef2 = (
                tf.sqrt(self.alphas) * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
            )
            self.posterior_variance = (
                self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
            )
    '''
    def get_prior_sample(self, batch_size):
        noise = tf.random.normal([batch_size, tf.shape(self.user_activity)[0]])
                
        if self.prior_type == 'noise':
            # 완전 무작위 노이즈 (Mean=0)
            return noise
        else:
            # 인기 기반 노이즈 (Mean=user_activity)
            return noise + tf.expand_dims(self.user_activity, 0)
    
    def q_sample(self, x_start, t):
        """Forward Process: x_0 -> x_t"""
        sqrt_alpha_bar_t = tf.gather(self.sqrt_alphas_cumprod, t)
        sqrt_one_minus_alpha_bar_t = tf.gather(self.sqrt_one_minus_alphas_cumprod, t)
        
        sqrt_alpha_bar_t = tf.reshape(sqrt_alpha_bar_t, [-1, 1])
        sqrt_one_minus_alpha_bar_t = tf.reshape(sqrt_one_minus_alpha_bar_t, [-1, 1])
        
        noise = tf.random.normal(shape=tf.shape(x_start))
        return sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise

    def q_posterior(self, x_start_pred, x_t, t):
        """Reverse Process의 Mean/Var 계산"""
        coef1 = tf.gather(self.posterior_mean_coef1, t)
        coef2 = tf.gather(self.posterior_mean_coef2, t)
        coef1 = tf.reshape(coef1, [-1, 1])
        coef2 = tf.reshape(coef2, [-1, 1])
        
        posterior_mean = coef1 * x_start_pred + coef2 * x_t
        posterior_log_variance = tf.gather(tf.math.log(tf.maximum(self.posterior_variance, 1e-20)), t)
        posterior_log_variance = tf.reshape(posterior_log_variance, [-1, 1])
        
        return posterior_mean, posterior_log_variance

    def p_sample(self, model, x_t, t_index, cond):
        """Reverse Step: x_t -> x_{t-1}"""
        batch_size = tf.shape(x_t)[0]
        t_tensor = tf.fill([batch_size], t_index)
        t_float = tf.cast(t_tensor, tf.float32) / float(self.steps)
        t_input = tf.reshape(t_float, [batch_size, 1])
        
        # 모델이 원본 데이터(x_start)를 직접 예측
        x_start_pred = model(x_t, cond, t_input, training=False)
        x_start_pred = tf.cast(x_start_pred, tf.float32)  # FP16 안전: mixed precision 시에도 float32 보장
        x_start_pred = tf.clip_by_value(x_start_pred, 0.0, 1.0) 

        model_mean, model_log_variance = self.q_posterior(x_start_pred, x_t, t_tensor)
        
        noise = tf.random.normal(shape=tf.shape(x_t))
        nonzero_mask = tf.cast(tf.reshape(t_tensor > 0, [-1, 1]), tf.float32)
        
        return model_mean + nonzero_mask * tf.exp(0.5 * model_log_variance) * noise