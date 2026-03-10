"""
=============================================================================
plot_efficiency.py  –  Flow Matching vs DDPM 추론 효율성 비교
=============================================================================
훈련 step(N)은 300으로 고정. 추론 step만 변화시켜 FM의 유연성을 검증.

Phase 1: Grid Search (N=300 고정)
  - lr ∈ {0.005, 0.001}, t_emb ∈ {32, 64}, dropout ∈ {0.0, 0.1, 0.2}
  → 모델별 최적 하이퍼파라미터 선정

Phase 2: Inference Step Sweep (최적 HP, N=300 훈련)
  - FM:   inference step ∈ {1..10, 20..100, 200, 300} 각각 평가
  - DDPM: inference step = 300 고정 (단일 점)
  - 5회 반복 → 평균/표준편차

사용법:
    python plot_efficiency.py
    python plot_efficiency.py --quick
    python plot_efficiency.py --datasets ML1M CiteULike

결과:  results/<dataset>/sweep_results.json
=============================================================================
"""
import yaml, os, json, time, argparse, sys, subprocess
import numpy as np

# ═══════════════════════════════════════════════════════════
# 인자 파싱
# ═══════════════════════════════════════════════════════════
parser = argparse.ArgumentParser()
parser.add_argument('--datasets', nargs='+', type=str, default=['ML1M', 'CiteULike'])
parser.add_argument('--train_N', type=int, default=300,
                    help='훈련 시 고정 time step (기본: 300)')
parser.add_argument('--infer_steps', nargs='+', type=int,
                    default=[1,2,5,6,7,10, 100,200,300])
parser.add_argument('--num_runs', type=int, default=5)
parser.add_argument('--prior_type', type=str, default='noise')
parser.add_argument('--inference_repeats', type=int, default=5)
parser.add_argument('--seed', type=int, default=2026)
parser.add_argument('--quick', action='store_true')
parser.add_argument('--epochs', type=int, default=None)
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--batch_size', type=int, default=None)

# subprocess 내부용
parser.add_argument('--_run_single', action='store_true', help=argparse.SUPPRESS)
parser.add_argument('--_model_type', type=str, help=argparse.SUPPRESS)
parser.add_argument('--_lr', type=float, help=argparse.SUPPRESS)
parser.add_argument('--_t_emb', type=int, help=argparse.SUPPRESS)
parser.add_argument('--_dropout', type=float, help=argparse.SUPPRESS)
parser.add_argument('--_dataset', type=str, help=argparse.SUPPRESS)
parser.add_argument('--_output_json', type=str, help=argparse.SUPPRESS)
parser.add_argument('--_mode', type=str, help=argparse.SUPPRESS)  # 'grid' or 'infer_sweep'
parser.add_argument('--_run_idx', type=int, help=argparse.SUPPRESS)
args = parser.parse_args()

# ═══════════════════════════════════════════════════════════
# 유틸리티
# ═══════════════════════════════════════════════════════════
def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def load_config(path="config.yaml"):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def fmt_time(seconds):
    if seconds < 60: return f"{seconds:.0f}s"
    elif seconds < 3600: return f"{seconds/60:.1f}m"
    else: return f"{seconds/3600:.1f}h"


# ═══════════════════════════════════════════════════════════
# SUBPROCESS WORKER
# ═══════════════════════════════════════════════════════════
def run_single_task():
    import tensorflow as tf
    import random, glob

    # ── GPU 최적화 ──
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    use_fp16 = args.fp16
    if use_fp16:
        try:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
        except Exception:
            use_fp16 = False

    try:
        tf.config.optimizer.set_jit(True)
    except Exception:
        pass

    from src.data_loader import ColdStartDataLoader
    from src.model import FlowModel
    from src.flow_logic import BernoulliFlow
    from src_ddpm.diffusion_logic import GaussianDiffusion

    mode = args._mode
    model_type = args._model_type
    N = args.train_N  # 항상 300
    lr = args._lr
    t_emb = args._t_emb
    dropout = args._dropout
    output_path = args._output_json

    config = load_config()
    config['dataset'] = args._dataset
    if args.batch_size:
        config['batch_size'] = args.batch_size

    if args.quick:
        epochs = 100
        patience_limit = 5
        eval_step = 5
    else:
        epochs = args.epochs or config.get('epochs', 500)
        patience_limit = config.get('patience', 10)
        eval_step = config.get('eval_step', 10)

    # 데이터 로드
    loader = ColdStartDataLoader(config)
    num_items, num_users = loader.build()
    train_ds = loader.get_dataset('train')
    vali_ds = loader.get_dataset('vali')
    test_ds = loader.get_dataset('test')

    try:
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
        vali_ds = vali_ds.prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    except Exception:
        pass

    log(f"  [{model_type.upper()} N={N} lr={lr} t_emb={t_emb} drop={dropout}] "
        f"Data: {num_items}x{num_users} | mode={mode}")

    # ── 벡터화 메트릭 계산 ──
    def calc_metrics_batch(pred_T, target_T, n_users, k_list):
        max_k = max(k_list)
        gt_mask = target_T > 0.5
        gt_counts = gt_mask.sum(axis=1)
        valid_mask = gt_counts > 0
        if valid_mask.sum() == 0:
            return {f'{m}@{k}': 0.0 for m in ['R', 'N', 'P', 'H'] for k in k_list}
        pred_valid = pred_T[valid_mask]
        gt_valid = gt_mask[valid_mask]
        gt_cnt = gt_counts[valid_mask]
        n_items = pred_valid.shape[1]
        if max_k < n_items:
            top_k_idx = np.argpartition(pred_valid, -max_k, axis=1)[:, -max_k:]
            rows = np.arange(pred_valid.shape[0])[:, None]
            top_k_scores = pred_valid[rows, top_k_idx]
            sorted_within = np.argsort(-top_k_scores, axis=1)
            top_k_idx = top_k_idx[rows, sorted_within]
        else:
            top_k_idx = np.argsort(-pred_valid, axis=1)[:, :max_k]
        rows = np.arange(top_k_idx.shape[0])[:, None]
        hits = gt_valid[rows, top_k_idx]
        result = {}
        for k in k_list:
            hits_at_k = hits[:, :k]
            n_hits = hits_at_k.sum(axis=1).astype(np.float64)
            result[f'R@{k}'] = float((n_hits / np.maximum(gt_cnt, 1).astype(np.float64)).mean())
            result[f'P@{k}'] = float((n_hits / k).mean())
            result[f'H@{k}'] = float(((n_hits > 0).astype(np.float64)).mean())
            log_pos = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))
            dcg = (hits_at_k.astype(np.float64) * log_pos[None, :]).sum(axis=1)
            cumsum_log = np.cumsum(log_pos)
            ideal_k = np.minimum(gt_cnt.astype(int), k)
            idcg = np.where(ideal_k > 0, cumsum_log[np.clip(ideal_k - 1, 0, k - 1)], 0.0)
            result[f'N@{k}'] = float((dcg / np.maximum(idcg, 1e-12)).mean())
        return result

    # ── FM: 고정 step 평가 ──
    def evaluate_flow_fixed(model, flow, dataset, N_steps, fixed_step, k_list=[10, 20]):
        if fixed_step <= 0:
            fixed_step = 1
        dt = 1.0 / N_steps
        outputs, targets = [], []
        for x_1, cond in dataset:
            bs = tf.shape(x_1)[0]
            targets.append(x_1.numpy())
            curr_x = flow.get_prior_sample(bs)
            for i in range(fixed_step):
                t_val = i * dt
                t_t = tf.fill([bs, 1], float(t_val))
                pred = model(curr_x, cond, t_t, training=False)
                curr_x = flow.inference_step(curr_x, pred, t_val, dt)
            # curr_x = ODE 적분 결과 (pred가 아님!)
            outputs.append(tf.cast(curr_x, tf.float32).numpy())
        target_T = np.concatenate(targets, axis=0).T
        pred_T = np.concatenate(outputs, axis=0).T
        return calc_metrics_batch(pred_T, target_T, target_T.shape[0], k_list)

    # ── FM: validation에서 best step 탐색 (스트리밍) ──
    def find_best_infer_step(model, flow, dataset, N_steps, k_list=[10, 20]):
        dt = 1.0 / N_steps
        all_targets = []
        batch_states = []
        for x_1, cond in dataset:
            all_targets.append(x_1.numpy())
            bs = tf.shape(x_1)[0]
            curr_x = flow.get_prior_sample(bs)
            batch_states.append({'curr_x': curr_x, 'cond': cond})
        target_T = np.concatenate(all_targets, axis=0).T
        n_u = target_T.shape[0]
        del all_targets
        best_step, best_r20 = 1, -1.0
        for i in range(N_steps):
            t_val = i * dt
            step_results = []
            for state in batch_states:
                bs = tf.shape(state['curr_x'])[0]
                t_t = tf.fill([bs, 1], float(t_val))
                pred = model(state['curr_x'], state['cond'], t_t, training=False)
                state['curr_x'] = flow.inference_step(state['curr_x'], pred, t_val, dt)
                # ODE 적분 결과(curr_x)로 평가 (pred 아님!)
                step_results.append(tf.cast(state['curr_x'], tf.float32).numpy())
            pred_T = np.concatenate(step_results, axis=0).T
            res = calc_metrics_batch(pred_T, target_T, n_u, k_list)
            if res['R@20'] > best_r20:
                best_r20 = res['R@20']
                best_step = i + 1
            del step_results, pred_T
        del batch_states, target_T
        return best_step

    # ── DDPM 평가 ──
    def evaluate_ddpm(model, diffusion, dataset, total_steps, k_list=[10, 20]):
        outputs, targets = [], []
        for x_true, cond in dataset:
            bs = tf.shape(x_true)[0]
            targets.append(x_true.numpy())
            x_t = diffusion.get_prior_sample(bs)
            for t in range(total_steps - 1, -1, -1):
                x_t = diffusion.p_sample(model, x_t, t, cond)
            outputs.append(tf.cast(x_t, tf.float32).numpy())
        target_T = np.concatenate(targets, axis=0).T
        pred_T = np.concatenate(outputs, axis=0).T
        return calc_metrics_batch(pred_T, target_T, target_T.shape[0], k_list)

    # ── 추론 시간 측정 ──
    def measure_time_flow(model, flow, dataset, N_steps, fixed_step, repeats=5):
        dt = 1.0 / N_steps
        for x_1, cond in dataset:
            bs = tf.shape(x_1)[0]
            curr_x = flow.get_prior_sample(bs)
            for i in range(fixed_step):
                t_t = tf.fill([bs, 1], float(i * dt))
                pred = model(curr_x, cond, t_t, training=False)
                curr_x = flow.inference_step(curr_x, pred, i * dt, dt)
            break
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            for x_1, cond in dataset:
                bs = tf.shape(x_1)[0]
                curr_x = flow.get_prior_sample(bs)
                for i in range(fixed_step):
                    t_t = tf.fill([bs, 1], float(i * dt))
                    pred = model(curr_x, cond, t_t, training=False)
                    curr_x = flow.inference_step(curr_x, pred, i * dt, dt)
            times.append(time.perf_counter() - t0)
        return float(np.mean(times)), float(np.std(times))

    def measure_time_ddpm(model, diffusion, dataset, total_steps, repeats=5):
        for x_true, cond in dataset:
            bs = tf.shape(x_true)[0]
            x_t = diffusion.get_prior_sample(bs)
            for t in range(total_steps - 1, -1, -1):
                x_t = diffusion.p_sample(model, x_t, t, cond)
            break
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            for x_true, cond in dataset:
                bs = tf.shape(x_true)[0]
                x_t = diffusion.get_prior_sample(bs)
                for t in range(total_steps - 1, -1, -1):
                    x_t = diffusion.p_sample(model, x_t, t, cond)
            times.append(time.perf_counter() - t0)
        return float(np.mean(times)), float(np.std(times))

    # ═══════════════════════════════════════════════════
    # 학습 + 평가
    # ═══════════════════════════════════════════════════
    os.makedirs("saved_model", exist_ok=True)
    random.seed(args.seed + (args._run_idx or 0))

    if model_type == 'flow':
        flow = BernoulliFlow(loader.user_activity, prior_type=args.prior_type)
        model = FlowModel(config['dims_mlp'] + [num_users], t_emb, dropout)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        if use_fp16:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

        @tf.function
        def step_fn_flow(x_1, cond, t, x_0):
            mask = tf.cast(tf.random.uniform(tf.shape(x_1)) < t, tf.float32)
            x_t = mask * x_1 + (1.0 - mask) * x_0
            with tf.GradientTape() as tape:
                pred = model(x_t, cond, t, training=True)
                loss = tf.reduce_mean(tf.square(tf.cast(x_1, tf.float32) - tf.cast(pred, tf.float32)))
                if use_fp16:
                    scaled_loss = optimizer.get_scaled_loss(loss)
            if use_fp16:
                scaled_grads = tape.gradient(scaled_loss, model.trainable_variables)
                grads = optimizer.get_unscaled_gradients(scaled_grads)
            else:
                grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return loss

        # ── 학습 ──
        best_r20, patience_cnt = -1.0, 0
        save_path = f"saved_model/_flow_N{N}_run{args._run_idx}"
        run_start = time.time()

        for epoch in range(epochs):
            ep_loss, ep_steps = 0.0, 0
            for x_1, cond in train_ds:
                bs = tf.shape(x_1)[0]
                t = tf.cast(tf.random.uniform((bs, 1), 1, N + 1, dtype=tf.int32), tf.float32) / N
                x_0 = flow.get_prior_sample(bs)
                loss = step_fn_flow(x_1, cond, t, x_0)
                ep_loss += loss.numpy()
                ep_steps += 1
            avg_loss = ep_loss / max(ep_steps, 1)

            if (epoch + 1) % eval_step == 0:
                val = evaluate_flow_fixed(model, flow, vali_ds, N, N)  # vali는 full step
                r20 = val['R@20']
                marker = ""
                if r20 > best_r20:
                    best_r20 = r20
                    patience_cnt = 0
                    model.save_weights(save_path)
                    marker = " ★"
                else:
                    patience_cnt += 1
                log(f"    E{epoch+1:03d}/{epochs} | Loss={avg_loss:.4f} | "
                    f"R@20={r20:.4f} | Best={best_r20:.4f} | "
                    f"pat={patience_cnt}/{patience_limit}{marker}")
                if patience_cnt >= patience_limit:
                    log(f"    → Early Stop at epoch {epoch+1}")
                    break

        try: model.load_weights(save_path)
        except: pass

        if mode == 'grid':
            # Grid: validation best step 탐색 후 R@20 반환
            best_step = find_best_infer_step(model, flow, vali_ds, N)
            val_res = evaluate_flow_fixed(model, flow, vali_ds, N, best_step)
            val_res['optimal_infer_step'] = best_step
            result = val_res

        elif mode == 'infer_sweep':
            # Sweep: 각 inference step에서 test set 평가
            sweep_steps = args.infer_steps
            all_step_results = {}

            for s in sweep_steps:
                test_res = evaluate_flow_fixed(model, flow, test_ds, N, s)
                t_mean, t_std = measure_time_flow(model, flow, test_ds, N, s,
                                                   repeats=args.inference_repeats)
                test_res['infer_time_mean'] = t_mean
                test_res['infer_time_std'] = t_std
                test_res['infer_step'] = s
                all_step_results[str(s)] = test_res
                log(f"    Step={s:>3d} | R@20={test_res['R@20']:.4f} | Time={t_mean:.3f}s")

            result = all_step_results

        # Cleanup
        for f in glob.glob(f"{save_path}*"):
            try: os.remove(f)
            except: pass

    elif model_type == 'ddpm':
        diffusion = GaussianDiffusion(loader.user_activity, steps=N, prior_type=args.prior_type)
        model = FlowModel(config['dims_mlp'] + [num_users], t_emb, dropout)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        if use_fp16:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

        @tf.function
        def step_fn_ddpm(x_start, cond, total_steps):
            batch_size = tf.shape(x_start)[0]
            t = tf.random.uniform([batch_size], minval=0, maxval=total_steps, dtype=tf.int32)
            x_t = diffusion.q_sample(x_start, t)
            t_input = tf.cast(t, tf.float32) / float(total_steps)
            t_input = tf.reshape(t_input, [batch_size, 1])
            with tf.GradientTape() as tape:
                pred = model(x_t, cond, t_input, training=True)
                loss = tf.reduce_mean(tf.square(tf.cast(x_start, tf.float32) - tf.cast(pred, tf.float32)))
                if use_fp16:
                    scaled_loss = optimizer.get_scaled_loss(loss)
            if use_fp16:
                scaled_grads = tape.gradient(scaled_loss, model.trainable_variables)
                grads = optimizer.get_unscaled_gradients(scaled_grads)
            else:
                grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return loss

        # ── 학습 ──
        best_r20, patience_cnt = -1.0, 0
        save_path = f"saved_model/_ddpm_N{N}_run{args._run_idx}"
        run_start = time.time()

        for epoch in range(epochs):
            ep_loss, ep_steps = 0.0, 0
            for x_start, cond in train_ds:
                loss = step_fn_ddpm(x_start, cond, N)
                ep_loss += loss.numpy()
                ep_steps += 1
            avg_loss = ep_loss / max(ep_steps, 1)

            if (epoch + 1) % eval_step == 0:
                val = evaluate_ddpm(model, diffusion, vali_ds, N)
                r20 = val['R@20']
                marker = ""
                if r20 > best_r20:
                    best_r20 = r20
                    patience_cnt = 0
                    model.save_weights(save_path)
                    marker = " ★"
                else:
                    patience_cnt += 1
                log(f"    E{epoch+1:03d}/{epochs} | Loss={avg_loss:.4f} | "
                    f"R@20={r20:.4f} | Best={best_r20:.4f} | "
                    f"pat={patience_cnt}/{patience_limit}{marker}")
                if patience_cnt >= patience_limit:
                    log(f"    → Early Stop at epoch {epoch+1}")
                    break

        try: model.load_weights(save_path)
        except: pass

        if mode == 'grid':
            val_res = evaluate_ddpm(model, diffusion, vali_ds, N)
            val_res['optimal_infer_step'] = N
            result = val_res

        elif mode == 'infer_sweep':
            test_res = evaluate_ddpm(model, diffusion, test_ds, N)
            t_mean, t_std = measure_time_ddpm(model, diffusion, test_ds, N,
                                               repeats=args.inference_repeats)
            test_res['infer_time_mean'] = t_mean
            test_res['infer_time_std'] = t_std
            test_res['infer_step'] = N
            result = {str(N): test_res}

        for f in glob.glob(f"{save_path}*"):
            try: os.remove(f)
            except: pass

    # 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log(f"  결과 저장 → {output_path}")


# ═══════════════════════════════════════════════════════════
# subprocess 실행 헬퍼
# ═══════════════════════════════════════════════════════════
def run_subprocess(mode, model_type, lr, t_emb, dropout, dataset, output_json, run_idx=0):
    cmd = [
        sys.executable, __file__,
        '--_run_single',
        '--_mode', mode,
        '--_model_type', model_type,
        '--_lr', str(lr),
        '--_t_emb', str(t_emb),
        '--_dropout', str(dropout),
        '--_dataset', dataset,
        '--_output_json', output_json,
        '--_run_idx', str(run_idx),
        '--train_N', str(args.train_N),
        '--prior_type', args.prior_type,
        '--inference_repeats', str(args.inference_repeats),
        '--seed', str(args.seed),
        '--infer_steps'] + [str(s) for s in args.infer_steps]
    if args.quick: cmd += ['--quick']
    if args.epochs: cmd += ['--epochs', str(args.epochs)]
    if args.fp16: cmd += ['--fp16']
    if args.batch_size: cmd += ['--batch_size', str(args.batch_size)]

    proc = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
    return proc.returncode


# ═══════════════════════════════════════════════════════════
# PHASE 1: Grid Search (동일)
# ═══════════════════════════════════════════════════════════
def grid_search(dataset):
    lr_list = [0.0005, 0.001, 0.0005]
    t_emb_list = [10, 20, 32, 64]
    dropout_list = [0.0, 0.1, 0.2]

    grid_dir = f"results/{dataset}/_grid"
    os.makedirs(grid_dir, exist_ok=True)
    best_hp = {}

    for model_type, model_label in [('flow', 'FlowMatching'), ('ddpm', 'DDPM')]:
        log(f"\n  Grid Search: {model_label} on {dataset} (N={args.train_N})")
        best_r20, best_params = -1.0, {'lr': lr_list[0], 't_emb': t_emb_list[0], 'dropout': dropout_list[0]}
        combo_idx, total = 0, len(lr_list) * len(t_emb_list) * len(dropout_list)

        for lr_val in lr_list:
            for t_emb_val in t_emb_list:
                for drop_val in dropout_list:
                    combo_idx += 1
                    temp_json = os.path.join(grid_dir, f"{model_type}_lr{lr_val}_emb{t_emb_val}_drop{drop_val}.json")
                    log(f"  [{combo_idx}/{total}] {model_label} | lr={lr_val}, t_emb={t_emb_val}, drop={drop_val}")

                    if os.path.exists(temp_json):
                        try:
                            with open(temp_json, 'r') as f:
                                result = json.load(f)
                            r20 = result.get('R@20', 0)
                            log(f"    → 기존: R@20={r20:.4f}")
                            if r20 > best_r20:
                                best_r20, best_params = r20, {'lr': lr_val, 't_emb': t_emb_val, 'dropout': drop_val}
                            continue
                        except: pass

                    ret = run_subprocess('grid', model_type, lr_val, t_emb_val, drop_val, dataset, temp_json)
                    if ret != 0:
                        log(f"    ❌ 실패"); continue
                    try:
                        with open(temp_json, 'r') as f:
                            result = json.load(f)
                        r20 = result.get('R@20', 0)
                        log(f"    → R@20={r20:.4f}")
                        if r20 > best_r20:
                            best_r20, best_params = r20, {'lr': lr_val, 't_emb': t_emb_val, 'dropout': drop_val}
                    except Exception as e:
                        log(f"    ❌ {e}")

        best_hp[model_label] = best_params
        log(f"  ✅ {model_label} 최적: {best_params} (R@20={best_r20:.4f})")

    hp_path = f"results/{dataset}/best_hyperparams.json"
    with open(hp_path, 'w', encoding='utf-8') as f:
        json.dump(best_hp, f, indent=2, ensure_ascii=False)
    return best_hp


# ═══════════════════════════════════════════════════════════
# PHASE 2: Inference Step Sweep
# ═══════════════════════════════════════════════════════════
def inference_sweep(dataset, best_hp):
    results_dir = f"results/{dataset}"
    temp_dir = f"results/{dataset}/_temp"
    os.makedirs(temp_dir, exist_ok=True)

    num_runs = 2 if args.quick else args.num_runs
    all_records = []  # [{model, infer_step, R@20, infer_time_mean, ...}, ...]

    for model_type, model_label in [('flow', 'FlowMatching'), ('ddpm', 'DDPM')]:
        hp = best_hp[model_label]
        log(f"\n{'='*50}")
        log(f"  {model_label}: 훈련 N={args.train_N}, HP={hp}, {num_runs} runs")
        log(f"{'='*50}")

        run_results = []  # 각 run의 {step: {metrics}} dict 리스트

        for run in range(num_runs):
            log(f"\n  ▶ {model_label} Run {run+1}/{num_runs}")
            temp_json = os.path.join(temp_dir, f"{model_type}_run{run}.json")

            ret = run_subprocess('infer_sweep', model_type,
                                 hp['lr'], hp['t_emb'], hp['dropout'],
                                 dataset, temp_json, run_idx=run)
            if ret != 0:
                log(f"    ❌ Run {run+1} 실패"); continue

            try:
                with open(temp_json, 'r') as f:
                    run_result = json.load(f)
                run_results.append(run_result)
                # 요약 출력
                if model_type == 'flow':
                    max_step = str(max(int(k) for k in run_result.keys()))
                    log(f"    ✓ {len(run_result)} steps 평가 완료 | "
                        f"Step={max_step} R@20={run_result[max_step]['R@20']:.4f}")
                else:
                    s = list(run_result.values())[0]
                    log(f"    ✓ R@20={s['R@20']:.4f} | Time={s['infer_time_mean']:.3f}s")
            except Exception as e:
                log(f"    ❌ 결과 읽기 실패: {e}")

        if not run_results:
            continue

        # 평균 계산: step별로
        all_steps = sorted(set(k for r in run_results for k in r.keys()), key=lambda x: int(x))

        for step_str in all_steps:
            step_data = [r[step_str] for r in run_results if step_str in r]
            if not step_data:
                continue
            avg = {}
            for key in step_data[0]:
                vals = [d[key] for d in step_data]
                try:
                    avg[key] = float(np.mean(vals))
                    if len(step_data) > 1:
                        avg[f'{key}_std'] = float(np.std(vals))
                except (TypeError, ValueError):
                    avg[key] = vals[0]

            avg['model'] = model_label
            avg['N'] = args.train_N
            avg['infer_step'] = int(step_str)
            avg['optimal_infer_step'] = int(step_str)  # result.py 호환
            all_records.append(avg)

    # 저장
    save_path = os.path.join(results_dir, "sweep_results.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump({
            'dataset': dataset,
            'train_N': args.train_N,
            'prior_type': args.prior_type,
            'num_runs': num_runs,
            'seed': args.seed,
            'best_hp': best_hp,
            'records': all_records
        }, f, indent=2, ensure_ascii=False)
    log(f"  💾 → {save_path}")

    # temp 정리
    import shutil
    try: shutil.rmtree(temp_dir)
    except: pass

    return all_records


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    log(f"{'='*60}")
    log(f"  FM vs DDPM — 추론 효율성 비교")
    log(f"  훈련 N: {args.train_N} (고정)")
    log(f"  추론 step sweep: {args.infer_steps}")
    log(f"  Datasets: {args.datasets}")
    log(f"  Runs: {2 if args.quick else args.num_runs}")
    log(f"  FP16: {'ON' if args.fp16 else 'OFF'}")
    log(f"{'='*60}")

    total_start = time.time()

    for dataset in args.datasets:
        log(f"\n{'#'*60}")
        log(f"  DATASET: {dataset}")
        log(f"{'#'*60}")

        results_dir = f"results/{dataset}"
        os.makedirs(results_dir, exist_ok=True)

        # Phase 1: Grid Search
        hp_path = f"results/{dataset}/best_hyperparams.json"
        if os.path.exists(hp_path):
            with open(hp_path, 'r') as f:
                best_hp = json.load(f)
            log(f"  기존 최적 HP: {best_hp}")
        else:
            best_hp = grid_search(dataset)

        # Phase 2: Inference Sweep
        records = inference_sweep(dataset, best_hp)

        # 요약
        log(f"\n  {dataset} 결과 요약:")
        for rec in records:
            log(f"    {rec['model']:>15} | InfStep={rec['infer_step']:>3d} | "
                f"R@20={rec.get('R@20',0):.4f} | Time={rec.get('infer_time_mean',0):.3f}s")

    log(f"\n✅ 완료! 총 {fmt_time(time.time()-total_start)}")
    log(f"다음: python result.py")


if __name__ == "__main__":
    if args._run_single:
        run_single_task()
    else:
        main()