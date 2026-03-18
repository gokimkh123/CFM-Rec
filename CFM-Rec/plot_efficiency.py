"""
=============================================================================
plot_efficiency.py  –  Flow Matching vs DDPM 공정 비교 실험
=============================================================================
Phase 1: Grid Search (N=300 고정)
  - lr ∈ {0.0005, 0.001}, t_emb ∈ {10, 20, 32, 64}, dropout ∈ {0.0, 0.1, 0.2}
  → 모델별 최적 HP 선정

Phase 2: Train N Sweep (최적 HP 사용)
  - sweep_steps: [1, 2, 5, 6, 7, 10, 100, 200, 300]
  - 각 N으로 FM/DDPM 훈련:
    - FM:   inference step 1~N (공차10)으로 vali에서 best S 탐색 → test 평가 → 1건 기록
    - DDPM: inference step = N 고정 → test 평가 → 1건 기록
  - 5회 반복 → 평균/표준편차

결과: results/<dataset>/sweep_results.json
  records: [{model, N, best_infer_step, R@10, R@20, N@10, N@20, infer_time_mean, ...}, ...]
=============================================================================
"""
import yaml, os, json, time, argparse, sys, subprocess
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', nargs='+', type=str, default=['ML1M', 'CiteULike'])
parser.add_argument('--sweep_steps', nargs='+', type=int,
                    default=[1, 2, 5, 6, 7, 10, 100, 200, 300])
parser.add_argument('--grid_N', type=int, default=300)
parser.add_argument('--infer_stride', type=int, default=10,
                    help='FM inference step 탐색 공차 (기본: 10)')
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
parser.add_argument('--_mode', type=str, help=argparse.SUPPRESS)
parser.add_argument('--_model_type', type=str, help=argparse.SUPPRESS)
parser.add_argument('--_train_N', type=int, help=argparse.SUPPRESS)
parser.add_argument('--_lr', type=float, help=argparse.SUPPRESS)
parser.add_argument('--_t_emb', type=int, help=argparse.SUPPRESS)
parser.add_argument('--_dropout', type=float, help=argparse.SUPPRESS)
parser.add_argument('--_dataset', type=str, help=argparse.SUPPRESS)
parser.add_argument('--_output_json', type=str, help=argparse.SUPPRESS)
parser.add_argument('--_run_idx', type=int, default=0, help=argparse.SUPPRESS)
args = parser.parse_args()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def load_config(path="config.yaml"):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def fmt_time(s):
    return f"{s:.0f}s" if s < 60 else f"{s/60:.1f}m" if s < 3600 else f"{s/3600:.1f}h"


# ═══════════════════════════════════════════════════════════
# SUBPROCESS WORKER: 1회 훈련 + 평가
# ═══════════════════════════════════════════════════════════
def run_single_task():
    import tensorflow as tf
    import random, glob

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
    N = args._train_N
    lr, t_emb, dropout = args._lr, args._t_emb, args._dropout
    output_path = args._output_json

    config = load_config()
    config['dataset'] = args._dataset
    if args.batch_size:
        config['batch_size'] = args.batch_size

    if args.quick:
        epochs, patience_limit, eval_step = 100, 5, 5
    else:
        epochs = args.epochs or config.get('epochs', 500)
        patience_limit = config.get('patience', 10)
        eval_step = config.get('eval_step', 10)

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

    # ── 벡터화 메트릭 ──
    def calc_metrics(pred_T, target_T, k_list=[10, 20]):
        max_k = max(k_list)
        gt_mask = target_T > 0.5
        gt_counts = gt_mask.sum(axis=1)
        valid = gt_counts > 0
        if valid.sum() == 0:
            return {f'{m}@{k}': 0.0 for m in ['R', 'N', 'P', 'H'] for k in k_list}
        pv, gv, gc = pred_T[valid], gt_mask[valid], gt_counts[valid]
        n_items = pv.shape[1]
        if max_k < n_items:
            idx = np.argpartition(pv, -max_k, axis=1)[:, -max_k:]
            rows = np.arange(pv.shape[0])[:, None]
            srt = np.argsort(-pv[rows, idx], axis=1)
            idx = idx[rows, srt]
        else:
            idx = np.argsort(-pv, axis=1)[:, :max_k]
        rows = np.arange(idx.shape[0])[:, None]
        hits = gv[rows, idx]
        result = {}
        for k in k_list:
            hk = hits[:, :k]
            nh = hk.sum(axis=1).astype(np.float64)
            result[f'R@{k}'] = float((nh / np.maximum(gc, 1).astype(np.float64)).mean())
            result[f'P@{k}'] = float((nh / k).mean())
            result[f'H@{k}'] = float(((nh > 0).astype(np.float64)).mean())
            log_pos = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))
            dcg = (hk.astype(np.float64) * log_pos[None, :]).sum(axis=1)
            cl = np.cumsum(log_pos)
            ik = np.minimum(gc.astype(int), k)
            idcg = np.where(ik > 0, cl[np.clip(ik - 1, 0, k - 1)], 0.0)
            result[f'N@{k}'] = float((dcg / np.maximum(idcg, 1e-12)).mean())
        return result

    # ── FM 평가 (curr_x 반환) ──
    def evaluate_flow(model, flow, dataset, N_steps, fixed_step):
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
            outputs.append(tf.cast(curr_x, tf.float32).numpy())
        target_T = np.concatenate(targets, axis=0).T
        pred_T = np.concatenate(outputs, axis=0).T
        return calc_metrics(pred_T, target_T)

    # ── DDPM 평가 ──
    def evaluate_ddpm(model, diffusion, dataset, total_steps):
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
        return calc_metrics(pred_T, target_T)

    # ── 추론 시간 측정 ──
    def measure_time_flow(model, flow, dataset, N_steps, fixed_step, repeats=5):
        dt = 1.0 / N_steps
        for x_1, cond in dataset:
            bs = tf.shape(x_1)[0]
            cx = flow.get_prior_sample(bs)
            for i in range(fixed_step):
                tt = tf.fill([bs, 1], float(i * dt))
                p = model(cx, cond, tt, training=False)
                cx = flow.inference_step(cx, p, i * dt, dt)
            break
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            for x_1, cond in dataset:
                bs = tf.shape(x_1)[0]
                cx = flow.get_prior_sample(bs)
                for i in range(fixed_step):
                    tt = tf.fill([bs, 1], float(i * dt))
                    p = model(cx, cond, tt, training=False)
                    cx = flow.inference_step(cx, p, i * dt, dt)
            times.append(time.perf_counter() - t0)
        return float(np.mean(times)), float(np.std(times))

    def measure_time_ddpm(model, diffusion, dataset, total_steps, repeats=5):
        for x_true, cond in dataset:
            bs = tf.shape(x_true)[0]
            xt = diffusion.get_prior_sample(bs)
            for t in range(total_steps - 1, -1, -1):
                xt = diffusion.p_sample(model, xt, t, cond)
            break
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            for x_true, cond in dataset:
                bs = tf.shape(x_true)[0]
                xt = diffusion.get_prior_sample(bs)
                for t in range(total_steps - 1, -1, -1):
                    xt = diffusion.p_sample(model, xt, t, cond)
            times.append(time.perf_counter() - t0)
        return float(np.mean(times)), float(np.std(times))

    # ═══════════════════════════════════════════════════
    # 학습 + 평가
    # ═══════════════════════════════════════════════════
    os.makedirs("saved_model", exist_ok=True)
    random.seed(args.seed + args._run_idx)

    if model_type == 'flow':
        flow = BernoulliFlow(loader.user_activity, prior_type=args.prior_type)
        model = FlowModel(config['dims_mlp'] + [num_users], t_emb, dropout)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        if use_fp16:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

        @tf.function
        def train_step(x_1, cond, t, x_0):
            mask = tf.cast(tf.random.uniform(tf.shape(x_1)) < t, tf.float32)
            x_t = mask * x_1 + (1.0 - mask) * x_0
            with tf.GradientTape() as tape:
                pred = model(x_t, cond, t, training=True)
                loss = tf.reduce_mean(tf.square(
                    tf.cast(x_1, tf.float32) - tf.cast(pred, tf.float32)))
                if use_fp16:
                    scaled_loss = optimizer.get_scaled_loss(loss)
            if use_fp16:
                grads = optimizer.get_unscaled_gradients(
                    tape.gradient(scaled_loss, model.trainable_variables))
            else:
                grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return loss

        # 학습 (vali는 full step으로 모델 품질 판단)
        best_r20, patience_cnt = -1.0, 0
        save_path = f"saved_model/_flow_N{N}_run{args._run_idx}"
        for epoch in range(epochs):
            ep_loss, ep_steps = 0.0, 0
            for x_1, cond in train_ds:
                bs = tf.shape(x_1)[0]
                t = tf.cast(tf.random.uniform((bs, 1), 1, N + 1, dtype=tf.int32),
                            tf.float32) / N
                x_0 = flow.get_prior_sample(bs)
                loss = train_step(x_1, cond, t, x_0)
                ep_loss += loss.numpy()
                ep_steps += 1

            if (epoch + 1) % eval_step == 0:
                val = evaluate_flow(model, flow, vali_ds, N, N)
                r20 = val['R@20']
                marker = ""
                if r20 > best_r20:
                    best_r20, patience_cnt = r20, 0
                    model.save_weights(save_path)
                    marker = " ★"
                else:
                    patience_cnt += 1
                log(f"    E{epoch+1:03d} | R@20={r20:.4f} | Best={best_r20:.4f} | "
                    f"pat={patience_cnt}/{patience_limit}{marker}")
                if patience_cnt >= patience_limit:
                    log(f"    → Early Stop")
                    break

        try:
            model.load_weights(save_path)
        except:
            pass

        if mode == 'grid':
            # Grid: vali R@20 반환
            result = evaluate_flow(model, flow, vali_ds, N, N)

        elif mode == 'sweep':
            # Sweep: vali에서 best S 탐색 → test에서 해당 S로 평가
            stride = args.infer_stride
            candidates = sorted(set([1] + list(range(1, N + 1, stride)) + [N]))
            log(f"    Best S 탐색: {candidates}")

            best_s, best_vali_r20 = N, -1.0
            for s in candidates:
                vali_res = evaluate_flow(model, flow, vali_ds, N, s)
                if vali_res['R@20'] > best_vali_r20:
                    best_vali_r20 = vali_res['R@20']
                    best_s = s
            log(f"    → Best S={best_s} (vali R@20={best_vali_r20:.4f})")

            # Test 평가 (best S)
            test_res = evaluate_flow(model, flow, test_ds, N, best_s)
            t_mean, t_std = measure_time_flow(model, flow, test_ds, N, best_s,
                                               repeats=args.inference_repeats)
            test_res['best_infer_step'] = best_s
            test_res['infer_time_mean'] = t_mean
            test_res['infer_time_std'] = t_std
            result = test_res

        for f in glob.glob(f"{save_path}*"):
            try: os.remove(f)
            except: pass

    elif model_type == 'ddpm':
        diffusion = GaussianDiffusion(loader.user_activity, steps=N,
                                       prior_type=args.prior_type)
        model = FlowModel(config['dims_mlp'] + [num_users], t_emb, dropout)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        if use_fp16:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

        @tf.function
        def train_step(x_start, cond, total_steps):
            batch_size = tf.shape(x_start)[0]
            t = tf.random.uniform([batch_size], minval=0, maxval=total_steps, dtype=tf.int32)
            x_t = diffusion.q_sample(x_start, t)
            t_input = tf.cast(t, tf.float32) / float(total_steps)
            t_input = tf.reshape(t_input, [batch_size, 1])
            with tf.GradientTape() as tape:
                pred = model(x_t, cond, t_input, training=True)
                loss = tf.reduce_mean(tf.square(
                    tf.cast(x_start, tf.float32) - tf.cast(pred, tf.float32)))
                if use_fp16:
                    scaled_loss = optimizer.get_scaled_loss(loss)
            if use_fp16:
                grads = optimizer.get_unscaled_gradients(
                    tape.gradient(scaled_loss, model.trainable_variables))
            else:
                grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return loss

        # 학습
        best_r20, patience_cnt = -1.0, 0
        save_path = f"saved_model/_ddpm_N{N}_run{args._run_idx}"
        for epoch in range(epochs):
            ep_loss, ep_steps = 0.0, 0
            for x_start, cond in train_ds:
                loss = train_step(x_start, cond, N)
                ep_loss += loss.numpy()
                ep_steps += 1

            if (epoch + 1) % eval_step == 0:
                val = evaluate_ddpm(model, diffusion, vali_ds, N)
                r20 = val['R@20']
                marker = ""
                if r20 > best_r20:
                    best_r20, patience_cnt = r20, 0
                    model.save_weights(save_path)
                    marker = " ★"
                else:
                    patience_cnt += 1
                log(f"    E{epoch+1:03d} | R@20={r20:.4f} | Best={best_r20:.4f} | "
                    f"pat={patience_cnt}/{patience_limit}{marker}")
                if patience_cnt >= patience_limit:
                    log(f"    → Early Stop")
                    break

        try:
            model.load_weights(save_path)
        except:
            pass

        if mode == 'grid':
            result = evaluate_ddpm(model, diffusion, vali_ds, N)

        elif mode == 'sweep':
            # DDPM: inference step = N 고정
            test_res = evaluate_ddpm(model, diffusion, test_ds, N)
            t_mean, t_std = measure_time_ddpm(model, diffusion, test_ds, N,
                                               repeats=args.inference_repeats)
            test_res['best_infer_step'] = N
            test_res['infer_time_mean'] = t_mean
            test_res['infer_time_std'] = t_std
            result = test_res

        for f in glob.glob(f"{save_path}*"):
            try: os.remove(f)
            except: pass

    # 저장
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    log(f"  ✓ 저장 → {output_path}")


# ═══════════════════════════════════════════════════════════
# subprocess 헬퍼
# ═══════════════════════════════════════════════════════════
def run_subprocess(mode, model_type, train_N, lr, t_emb, dropout,
                   dataset, output_json, run_idx=0):
    cmd = [
        sys.executable, __file__,
        '--_run_single',
        '--_mode', mode,
        '--_model_type', model_type,
        '--_train_N', str(train_N),
        '--_lr', str(lr), '--_t_emb', str(t_emb), '--_dropout', str(dropout),
        '--_dataset', dataset, '--_output_json', output_json,
        '--_run_idx', str(run_idx),
        '--prior_type', args.prior_type,
        '--inference_repeats', str(args.inference_repeats),
        '--seed', str(args.seed),
        '--infer_stride', str(args.infer_stride),
    ]
    if args.quick: cmd += ['--quick']
    if args.epochs: cmd += ['--epochs', str(args.epochs)]
    if args.fp16: cmd += ['--fp16']
    if args.batch_size: cmd += ['--batch_size', str(args.batch_size)]

    return subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr).returncode


# ═══════════════════════════════════════════════════════════
# PHASE 1: Grid Search (N=300)
# ═══════════════════════════════════════════════════════════
def grid_search(dataset):
    grid_N = args.grid_N
    lr_list = [0.0005, 0.001]
    t_emb_list = [32, 64]
    dropout_list = [0.0, 0.1]

    grid_dir = f"results/{dataset}/_grid"
    os.makedirs(grid_dir, exist_ok=True)
    best_hp = {}

    for model_type, label in [('flow', 'FlowMatching'), ('ddpm', 'DDPM')]:
        log(f"\n  Grid Search: {label} on {dataset} (N={grid_N})")
        best_r20, best_params = -1.0, {'lr': lr_list[0], 't_emb': t_emb_list[0], 'dropout': dropout_list[0]}
        total = len(lr_list) * len(t_emb_list) * len(dropout_list)
        ci = 0
        for lr in lr_list:
            for te in t_emb_list:
                for dr in dropout_list:
                    ci += 1
                    fpath = os.path.join(grid_dir, f"{model_type}_lr{lr}_emb{te}_drop{dr}.json")
                    log(f"  [{ci}/{total}] lr={lr}, t_emb={te}, drop={dr}")
                    if os.path.exists(fpath):
                        try:
                            with open(fpath) as f:
                                r20 = json.load(f).get('R@20', 0)
                            log(f"    → 기존: R@20={r20:.4f}")
                            if r20 > best_r20:
                                best_r20, best_params = r20, {'lr': lr, 't_emb': te, 'dropout': dr}
                            continue
                        except: pass
                    ret = run_subprocess('grid', model_type, grid_N, lr, te, dr, dataset, fpath)
                    if ret != 0:
                        log(f"    ❌ 실패"); continue
                    try:
                        with open(fpath) as f:
                            r20 = json.load(f).get('R@20', 0)
                        log(f"    → R@20={r20:.4f}")
                        if r20 > best_r20:
                            best_r20, best_params = r20, {'lr': lr, 't_emb': te, 'dropout': dr}
                    except Exception as e:
                        log(f"    ❌ {e}")

        best_hp[label] = best_params
        log(f"  ✅ {label} 최적: {best_params} (R@20={best_r20:.4f})")

    hp_path = f"results/{dataset}/best_hyperparams.json"
    with open(hp_path, 'w') as f:
        json.dump(best_hp, f, indent=2)
    log(f"  💾 → {hp_path}")
    return best_hp


# ═══════════════════════════════════════════════════════════
# PHASE 2: Train N Sweep
# ═══════════════════════════════════════════════════════════
def sweep(dataset, best_hp):
    results_dir = f"results/{dataset}"
    temp_dir = f"results/{dataset}/_temp"
    os.makedirs(temp_dir, exist_ok=True)

    num_runs = 2 if args.quick else args.num_runs
    all_records = []

    # 이어하기
    save_path = os.path.join(results_dir, "sweep_results.json")
    existing_keys = set()
    if os.path.exists(save_path):
        try:
            with open(save_path) as f:
                existing = json.load(f)
            all_records = existing.get('records', [])
            for r in all_records:
                existing_keys.add(f"{r['model']}_{r['N']}")
            log(f"  기존 {len(all_records)}건 로드")
        except: pass

    total_start = time.time()

    for idx, N in enumerate(args.sweep_steps):
        log(f"\n{'='*50}")
        log(f"  Train N = {N}  ({idx+1}/{len(args.sweep_steps)})")
        log(f"{'='*50}")

        for model_type, label in [('flow', 'FlowMatching'), ('ddpm', 'DDPM')]:
            key = f"{label}_{N}"
            if key in existing_keys:
                log(f"  ⏭ {label} N={N} — 건너뜀")
                continue

            hp = best_hp[label]
            log(f"\n▶ {label} (N={N}) HP={hp} | {num_runs} runs")

            run_results = []
            for run in range(num_runs):
                fpath = os.path.join(temp_dir, f"{model_type}_N{N}_run{run}.json")
                if os.path.exists(fpath):
                    try:
                        with open(fpath) as f:
                            run_results.append(json.load(f))
                        log(f"    Run {run+1}: 기존 로드")
                        continue
                    except: pass

                log(f"    Run {run+1}/{num_runs}")
                ret = run_subprocess('sweep', model_type, N,
                                     hp['lr'], hp['t_emb'], hp['dropout'],
                                     dataset, fpath, run_idx=run)
                if ret != 0:
                    log(f"    ❌ 실패"); continue
                try:
                    with open(fpath) as f:
                        run_results.append(json.load(f))
                except Exception as e:
                    log(f"    ❌ {e}")

            if not run_results:
                continue

            # 평균 계산
            avg = {'model': label, 'N': N}
            for key_name in run_results[0]:
                vals = [r[key_name] for r in run_results]
                try:
                    avg[key_name] = float(np.mean(vals))
                    if len(run_results) > 1:
                        avg[f'{key_name}_std'] = float(np.std(vals))
                except (TypeError, ValueError):
                    from collections import Counter
                    avg[key_name] = Counter(vals).most_common(1)[0][0]

            all_records.append(avg)
            existing_keys.add(f"{label}_{N}")
            log(f"  ✅ {label} N={N}: S={avg.get('best_infer_step','N/A')} | "
                f"R@20={avg.get('R@20',0):.4f}")

        # 중간 저장
        with open(save_path, 'w') as f:
            json.dump({
                'dataset': dataset,
                'prior_type': args.prior_type,
                'num_runs': num_runs,
                'best_hp': best_hp,
                'records': all_records
            }, f, indent=2)
        log(f"  💾 → {save_path} | 경과: {fmt_time(time.time()-total_start)}")

    import shutil
    try: shutil.rmtree(temp_dir)
    except: pass
    return all_records


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    log(f"{'='*60}")
    log(f"  Flow Matching vs DDPM 공정 비교")
    log(f"  Datasets: {args.datasets}")
    log(f"  Grid N={args.grid_N} | Sweep: {args.sweep_steps}")
    log(f"  FM infer stride: {args.infer_stride}")
    log(f"  Runs: {2 if args.quick else args.num_runs}")
    log(f"{'='*60}")

    for dataset in args.datasets:
        log(f"\n{'#'*60}")
        log(f"  DATASET: {dataset}")
        log(f"{'#'*60}")

        os.makedirs(f"results/{dataset}", exist_ok=True)

        hp_path = f"results/{dataset}/best_hyperparams.json"
        if os.path.exists(hp_path):
            with open(hp_path) as f:
                best_hp = json.load(f)
            log(f"  기존 HP 로드: {best_hp}")
        else:
            best_hp = grid_search(dataset)

        records = sweep(dataset, best_hp)

        log(f"\n  {dataset}: {len(records)} records 완료")
        for r in sorted(records, key=lambda x: (x['model'], x['N'])):
            log(f"    {r['model']:<15} N={r['N']:>3d} S={r.get('best_infer_step','?'):>3} "
                f"R@20={r.get('R@20',0):.4f}")

    log(f"\n✅ 완료! → python result.py")


if __name__ == "__main__":
    if args._run_single:
        run_single_task()
    else:
        main()