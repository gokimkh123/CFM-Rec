"""
=============================================================================
plot_fm_recall_curve.py  –  FM Inference Step vs Recall@20 곡선
=============================================================================
N=300으로 훈련된 FM 모델에서 추론 스텝을 1~300 (공차 5)으로 바꿔가며
Recall@20 성능을 측정하고, ML1M/CiteULike를 한 그래프에 표시.

사용법:
    python plot_fm_recall_curve.py
    python plot_fm_recall_curve.py --quick
    python plot_fm_recall_curve.py --num_runs 3

결과:
    results/fm_recall_curve.json
    results/fm_recall_curve.png
=============================================================================
"""
import yaml, os, json, time, argparse, sys, subprocess
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', nargs='+', type=str, default=['ML1M', 'CiteULike'])
parser.add_argument('--train_N', type=int, default=300)
parser.add_argument('--num_runs', type=int, default=5)
parser.add_argument('--prior_type', type=str, default='noise')
parser.add_argument('--seed', type=int, default=2026)
parser.add_argument('--quick', action='store_true')
parser.add_argument('--epochs', type=int, default=None)
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--batch_size', type=int, default=None)
parser.add_argument('--stride', type=int, default=5, help='Inference step 간격 (기본: 5)')

# subprocess 내부용
parser.add_argument('--_run_single', action='store_true', help=argparse.SUPPRESS)
parser.add_argument('--_lr', type=float, help=argparse.SUPPRESS)
parser.add_argument('--_t_emb', type=int, help=argparse.SUPPRESS)
parser.add_argument('--_dropout', type=float, help=argparse.SUPPRESS)
parser.add_argument('--_dataset', type=str, help=argparse.SUPPRESS)
parser.add_argument('--_output_json', type=str, help=argparse.SUPPRESS)
parser.add_argument('--_run_idx', type=int, help=argparse.SUPPRESS)
args = parser.parse_args()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def load_config(path="config.yaml"):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def fmt_time(s):
    return f"{s:.0f}s" if s < 60 else f"{s/60:.1f}m" if s < 3600 else f"{s/3600:.1f}h"


# ═══════════════════════════════════════════════════════════
# SUBPROCESS WORKER: FM 1회 학습 + 전체 inference step 평가
# ═══════════════════════════════════════════════════════════
def run_single():
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

    N = args.train_N
    lr, t_emb, dropout = args._lr, args._t_emb, args._dropout

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

    log(f"  [FM N={N} lr={lr} t_emb={t_emb} drop={dropout}] "
        f"Data: {num_items}x{num_users} | run={args._run_idx}")

    # ── 벡터화 메트릭 ──
    def calc_metrics(pred_T, target_T, k_list=[10, 20]):
        max_k = max(k_list)
        gt_mask = target_T > 0.5
        gt_counts = gt_mask.sum(axis=1)
        valid = gt_counts > 0
        if valid.sum() == 0:
            return {f'{m}@{k}': 0.0 for m in ['R', 'N'] for k in k_list}
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
            log_pos = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))
            dcg = (hk.astype(np.float64) * log_pos[None, :]).sum(axis=1)
            cl = np.cumsum(log_pos)
            ik = np.minimum(gc.astype(int), k)
            idcg = np.where(ik > 0, cl[np.clip(ik - 1, 0, k - 1)], 0.0)
            result[f'N@{k}'] = float((dcg / np.maximum(idcg, 1e-12)).mean())
        return result

    # ── FM 고정 step 평가 ──
    def evaluate_fixed(model, flow, dataset, N_steps, fixed_step):
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
        return calc_metrics(pred_T, target_T)

    # ── 학습 ──
    os.makedirs("saved_model", exist_ok=True)
    random.seed(args.seed + (args._run_idx or 0))

    flow = BernoulliFlow(loader.user_activity, prior_type=args.prior_type)
    model = FlowModel(config['dims_mlp'] + [num_users], t_emb, dropout)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    if use_fp16:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    @tf.function
    def step_fn(x_1, cond, t, x_0):
        mask = tf.cast(tf.random.uniform(tf.shape(x_1)) < t, tf.float32)
        x_t = mask * x_1 + (1.0 - mask) * x_0
        with tf.GradientTape() as tape:
            pred = model(x_t, cond, t, training=True)
            loss = tf.reduce_mean(tf.square(tf.cast(x_1, tf.float32) - tf.cast(pred, tf.float32)))
            if use_fp16:
                scaled_loss = optimizer.get_scaled_loss(loss)
        if use_fp16:
            grads = optimizer.get_unscaled_gradients(tape.gradient(scaled_loss, model.trainable_variables))
        else:
            grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    best_r20, patience_cnt = -1.0, 0
    save_path = f"saved_model/_fm_curve_run{args._run_idx}"

    for epoch in range(epochs):
        ep_loss, ep_steps = 0.0, 0
        for x_1, cond in train_ds:
            bs = tf.shape(x_1)[0]
            t = tf.cast(tf.random.uniform((bs, 1), 1, N + 1, dtype=tf.int32), tf.float32) / N
            x_0 = flow.get_prior_sample(bs)
            loss = step_fn(x_1, cond, t, x_0)
            ep_loss += loss.numpy()
            ep_steps += 1

        if (epoch + 1) % eval_step == 0:
            val = evaluate_fixed(model, flow, vali_ds, N, N)
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
    # ── 전체 inference step 평가 ──
    # 1, 1+stride, 1+2*stride, ..., 마지막에 N 포함
    stride = args.stride
    steps = sorted(set([1] + list(range(1, N + 1, stride)) + [N]))
    log(f"  평가 중: {len(steps)} steps (stride={stride})")

    results = {}
    for s in steps:
        start_time = time.time()  # 💡 [추가] 추론 시작 시간 측정
        res = evaluate_fixed(model, flow, test_ds, N, s)
        infer_time = time.time() - start_time  # 💡 [추가] 추론 소요 시간 계산
        
        res['infer_time'] = infer_time  # 💡 [추가] 평가 결과 딕셔너리에 시간 삽입!
        results[str(s)] = res
        
        if s % 50 == 0 or s <= 5 or s == steps[-1]:
            # 💡 [추가] 콘솔 출력에도 몇 초가 걸렸는지 표시하도록 수정
            log(f"    Step={s:>3d} | R@20={res['R@20']:.4f} | Time={infer_time:.3f}s")

    # 저장
    with open(args._output_json, 'w') as f:
        json.dump(results, f, indent=2)
    log(f"  ✓ 저장 → {args._output_json}")

    # 정리
    for fp in glob.glob(f"{save_path}*"):
        try:
            os.remove(fp)
        except:
            pass


# ═══════════════════════════════════════════════════════════
# MAIN: 데이터셋별 subprocess 관리 + 그래프 생성
# ═══════════════════════════════════════════════════════════
def run_subprocess(dataset, lr, t_emb, dropout, output_json, run_idx):
    cmd = [
        sys.executable, __file__,
        '--_run_single',
        '--_lr', str(lr), '--_t_emb', str(t_emb), '--_dropout', str(dropout),
        '--_dataset', dataset, '--_output_json', output_json,
        '--_run_idx', str(run_idx),
        '--train_N', str(args.train_N),
        '--num_runs', str(args.num_runs),
        '--prior_type', args.prior_type,
        '--seed', str(args.seed),
        '--stride', str(args.stride),
    ]
    if args.quick: cmd += ['--quick']
    if args.epochs: cmd += ['--epochs', str(args.epochs)]
    if args.fp16: cmd += ['--fp16']
    if args.batch_size: cmd += ['--batch_size', str(args.batch_size)]

    proc = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
    return proc.returncode


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    os.makedirs("results", exist_ok=True)
    num_runs = 2 if args.quick else args.num_runs

    log(f"{'='*60}")
    log(f"  FM Inference Step vs Recall@20 Curve")
    log(f"  Datasets: {args.datasets}")
    log(f"  Train N: {args.train_N}, Stride: {args.stride}")
    log(f"  Runs: {num_runs}")
    log(f"{'='*60}")

    all_data = {}  # {dataset: {step: {'R@20': mean, 'R@20_std': std, ...}}}

    for dataset in args.datasets:
        log(f"\n{'#'*50}")
        log(f"  DATASET: {dataset}")
        log(f"{'#'*50}")

        # Grid Search 결과에서 FM 최적 HP 로드
        hp_path = f"results/{dataset}/best_hyperparams.json"
        if os.path.exists(hp_path):
            with open(hp_path, 'r') as f:
                best_hp = json.load(f)
            hp = best_hp.get('FlowMatching', {'lr': 0.001, 't_emb': 32, 'dropout': 0.0})
            log(f"  HP 로드: {hp}")
        else:
            hp = {'lr': 0.001, 't_emb': 32, 'dropout': 0.0}
            log(f"  ⚠ best_hyperparams.json 없음, 기본 HP 사용: {hp}")

        temp_dir = f"results/{dataset}/_fm_curve_temp"
        os.makedirs(temp_dir, exist_ok=True)

        run_results = []  # [{step_str: {metrics}}, ...]

        for run in range(num_runs):
            log(f"\n  ▶ Run {run+1}/{num_runs}")
            temp_json = os.path.join(temp_dir, f"run{run}.json")

            # 이미 결과 있으면 스킵
            if os.path.exists(temp_json):
                try:
                    with open(temp_json, 'r') as f:
                        r = json.load(f)
                    run_results.append(r)
                    log(f"    → 기존 결과 로드 ({len(r)} steps)")
                    continue
                except:
                    pass

            ret = run_subprocess(dataset, hp['lr'], hp['t_emb'], hp['dropout'],
                                 temp_json, run)
            if ret != 0:
                log(f"    ❌ 실패")
                continue

            try:
                with open(temp_json, 'r') as f:
                    r = json.load(f)
                run_results.append(r)
                log(f"    ✓ {len(r)} steps 완료")
            except Exception as e:
                log(f"    ❌ 결과 읽기 실패: {e}")

        if not run_results:
            log(f"  ⚠ {dataset}: 결과 없음")
            continue

        # step별 평균 계산
        all_steps = sorted(set(k for r in run_results for k in r.keys()), key=lambda x: int(x))
        ds_data = {}
        for step_str in all_steps:
            step_vals = [r[step_str] for r in run_results if step_str in r]
            if not step_vals:
                continue
            avg = {}
            for key in step_vals[0]:
                vals = [d[key] for d in step_vals]
                try:
                    avg[key] = float(np.mean(vals))
                    if len(step_vals) > 1:
                        avg[f'{key}_std'] = float(np.std(vals))
                except (TypeError, ValueError):
                    avg[key] = vals[0]
            ds_data[step_str] = avg

        all_data[dataset] = ds_data
        log(f"  ✅ {dataset}: {len(ds_data)} steps 평균 완료")

    # ── 결과 저장 ──
    result_path = "results/fm_recall_curve.json"
    with open(result_path, 'w') as f:
        json.dump(all_data, f, indent=2)
    log(f"\n💾 결과 저장 → {result_path}")

    # ── 그래프 생성 ──
    if not all_data:
        log("⚠ 데이터 없음, 그래프 생성 스킵")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {
        'ML1M': '#FF9800',
        'CiteULike': '#4CAF50',
        'Books': '#4CAF50',
    }
    default_colors = ['#FF9800', '#9C27B0', '#795548']

    for i, (dataset, ds_data) in enumerate(all_data.items()):
        steps = sorted([int(k) for k in ds_data.keys()])
        r20_vals = [ds_data[str(s)]['R@20'] for s in steps]
        r20_stds = [ds_data[str(s)].get('R@20_std', 0) for s in steps]

        color = colors.get(dataset, default_colors[i % len(default_colors)])

        ax.plot(steps, r20_vals, 'o-', color=color, markersize=4,
                linewidth=2, label=dataset)
        '''
        # 표준편차 영역 (반투명)
        if any(s > 0 for s in r20_stds):
            r20_arr = np.array(r20_vals)
            std_arr = np.array(r20_stds)
            ax.fill_between(steps, r20_arr - std_arr, r20_arr + std_arr,
                            color=color, alpha=0.15)
        '''
    ax.set_xlabel('Inference Step', fontsize=20)
    ax.set_ylabel('Recall@20', fontsize=20)
    ax.set_title('CFM: Inference Step vs Recall@20 (N=300 fixed)',
                 fontsize=20, fontweight='bold')
    ax.legend(fontsize=15, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, args.train_N + 5)

    plt.tight_layout()
    fig_path = "results/fm_recall_curve.png"
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    log(f"📊 그래프 저장 → {fig_path}")

    # ── 정리 ──
    import shutil
    for dataset in args.datasets:
        temp_dir = f"results/{dataset}/_fm_curve_temp"
        # temp는 유지 (이어하기용)

    log(f"\n✅ 완료!")


if __name__ == "__main__":
    if args._run_single:
        run_single()
    else:
        main()