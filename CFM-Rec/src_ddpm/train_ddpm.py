# src_ddpm/train_ddpm.py
from logging import config

import tensorflow as tf
import yaml
import os
import glob
import numpy as np
import datetime
import argparse
import random
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn, TaskProgressColumn

from src.data_loader import ColdStartDataLoader
from src.model import FlowModel 
from src_ddpm.diffusion_logic import GaussianDiffusion
from src.metrics import compute_metrics

console = Console()

parser = argparse.ArgumentParser()
parser.add_argument('--steps', type=int, default=100, help='Total Diffusion Steps (N)')
parser.add_argument('--prior_type', type=str, default='noise')
args = parser.parse_args()

def load_config(path="config.yaml"):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def _calculate_metrics_batch(pred_matrix_T, target_matrix_T, num_users, k_list):
    """배치 단위 메트릭 계산 함수"""
    metrics_keys = ['R', 'N', 'P', 'H']
    raw_results = {f'{key}@{k}': [] for key in metrics_keys for k in k_list}
    
    for u in range(num_users):
        gt_items = np.where(target_matrix_T[u] > 0.5)[0]
        if len(gt_items) == 0: continue
        
        top_indices = np.argsort(pred_matrix_T[u])[-max(k_list):][::-1]
        m = compute_metrics(top_indices, gt_items, k_list=k_list)
        
        for k in k_list:
            raw_results[f'R@{k}'].append(m.get(f'Recall@{k}', 0.0))
            raw_results[f'N@{k}'].append(m.get(f'NDCG@{k}', 0.0))
            raw_results[f'P@{k}'].append(m.get(f'Precision@{k}', 0.0))
            raw_results[f'H@{k}'].append(m.get(f'Hit@{k}', 0.0))
            
    return {k: np.mean(v) if v else 0.0 for k, v in raw_results.items()}

def evaluate_ddpm(model, diffusion, dataset, total_steps, k_list=[10, 20]):
    """Search 과정 없이 지정된 total_steps만큼만 역확산 후 평가"""
    # 콘솔 출력 생략 (다중 런에서는 너무 많은 로그가 생성되므로)
    
    final_outputs = [] 
    all_targets = []
    
    for x_true, cond in dataset:
        batch_bs = tf.shape(x_true)[0]
        all_targets.append(x_true.numpy())
        
        # 1. Prior Sampling (Noise or Popularity)
        x_t = diffusion.get_prior_sample(batch_bs)
        
        # 2. Reverse Process Loop (T-1 -> 0)
        for t in range(total_steps - 1, -1, -1):
            x_t = diffusion.p_sample(model, x_t, t, cond)
            
        # 3. 최종 결과만 저장
        final_outputs.append(x_t.numpy())

    # 정답 및 예측 행렬 병합
    target_matrix = np.concatenate(all_targets, axis=0)
    pred_matrix = np.concatenate(final_outputs, axis=0)
    
    target_matrix_T = target_matrix.T
    pred_matrix_T = pred_matrix.T
    num_users = target_matrix_T.shape[0]
    
    # 메트릭 계산
    results = _calculate_metrics_batch(pred_matrix_T, target_matrix_T, num_users, k_list)
    results['Best_Step'] = total_steps
    return results

def train():
    title = "Popularity Prior" if args.prior_type == 'popularity' else "Pure Noise Prior"
    console.print(Panel.fit(f"[bold yellow]DDPM Training ({title}, Fixed Steps={args.steps})[/]", border_style="yellow"))

    config = load_config()
    dataset_name = config.get('dataset', 'ML1M')
    save_dir = f"saved_model_ddpm_{dataset_name}"
    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)

    # 이전 모델 가중치 파일 정리
    for f in glob.glob(f"{save_dir}/best_ddpm_model*"):
        try: os.remove(f)
        except OSError: pass
    
    # 1. 데이터셋 1회 로드 (메모리 적재)
    with console.status("[bold green]Loading Data...", spinner="dots"):
        loader = ColdStartDataLoader(config)
        num_items, num_users = loader.build()
        train_ds = loader.get_dataset(mode='train')
        vali_ds = loader.get_dataset(mode='vali') 
        test_ds = loader.get_dataset(mode='test')

    diffusion = GaussianDiffusion(loader.user_activity, steps=args.steps, prior_type=args.prior_type)

    epochs = config['epochs']
    eval_step = config.get('eval_step', 10)
    patience_limit = config.get('patience', 10)
    
    NUM_RUNS = 5
    all_test_results = []
    random.seed(2026)
    # 2. 5회 반복 실행 (Random Search)
    for run in range(NUM_RUNS):
        sampled_lr = random.choice([0.0005, 0.001])
        sampled_time_emb = random.choice([32, 64])
        sampled_dropout = random.choice([0.0, 0.1, 0.2])
        
        console.print(f"\n[bold magenta]=== Run {run+1}/{NUM_RUNS} | LR: {sampled_lr} | TimeEmb: {sampled_time_emb} | Dropout: {sampled_dropout} ===[/]")

        model_dims = config['dims_mlp'] + [num_users]
        model = FlowModel(model_dims, sampled_time_emb, sampled_dropout)
        optimizer = tf.keras.optimizers.Adam(learning_rate=sampled_lr)

        # 모델 초기화 후 내부 스코프에 tf.function 정의 (그래프 충돌 방지)
        @tf.function
        def step_fn(x_start, cond, total_steps):
            batch_size = tf.shape(x_start)[0]
            t = tf.random.uniform([batch_size], minval=0, maxval=total_steps, dtype=tf.int32)
            x_t = diffusion.q_sample(x_start, t)
            
            t_input = tf.cast(t, tf.float32) / float(total_steps)
            t_input = tf.reshape(t_input, [batch_size, 1])
            
            with tf.GradientTape() as tape:
                pred_x_start = model(x_t, cond, t_input, training=True)
                loss = tf.reduce_mean(tf.square(x_start - pred_x_start))
            
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return loss

        best_recall = -1.0
        patience_cnt = 0
        current_save_path = f"{save_dir}/best_ddpm_model_run{run}"

        progress = Progress(
            SpinnerColumn(), TextColumn("[bold blue]{task.description}"), BarColumn(),
            TaskProgressColumn(), TimeRemainingColumn(), TextColumn("{task.fields[info]}"), console=console
        )

        with progress:
            overall_task = progress.add_task(f"[bold magenta]Run {run+1} Progress", total=epochs, info="")
            epoch_task = progress.add_task("[cyan]Current Epoch", total=1, info="Loss: N/A")

            for epoch in range(epochs):
                progress.reset(epoch_task)
                progress.update(epoch_task, description=f"[cyan]Epoch {epoch+1}/{epochs}")
                
                # --- Train Phase ---
                train_loss, steps = 0, 0
                for x_start, cond in train_ds:
                    # 내부 step_fn 호출
                    loss = step_fn(x_start, cond, args.steps)
                    train_loss += loss.numpy()
                    steps += 1
                    progress.update(epoch_task, advance=1, total=steps, info=f"Loss: {loss.numpy():.4f}")
                
                avg_loss = train_loss / steps

                # --- Validation Phase ---
                if (epoch + 1) % eval_step == 0:
                    val_metrics = evaluate_ddpm(model, diffusion, vali_ds, args.steps)
                    r10, r20 = val_metrics['R@10'], val_metrics['R@20']
                    
                    log_msg = f"E{epoch+1:03d} | Loss: {avg_loss:.4f} | Val R@10: {r10:.4f} | Val R@20: {r20:.4f}"
                    
                    if r20 > best_recall:
                        best_recall = r20
                        patience_cnt = 0
                        model.save_weights(current_save_path)
                        log_msg += f" [bold green]★ Best[/]"
                    else:
                        patience_cnt += 1
                        
                    console.print(log_msg)
                    if patience_cnt >= patience_limit:
                        console.print("[bold red]Early Stopping Triggered.[/]")
                        break
                progress.update(overall_task, advance=1)

        # --- Test Phase (개별 Run) ---
        try: model.load_weights(current_save_path)
        except: pass
        
        test_metrics = evaluate_ddpm(model, diffusion, test_ds, args.steps)
        all_test_results.append(test_metrics)
        
        # 메모리 정리 (VRAM OOM 방지)
        tf.keras.backend.clear_session()

    # 3. 5회 실행 결과 평균 산출
    console.print(f"\n[bold yellow]=== Calculating Average over {NUM_RUNS} runs ===[/]")
    avg_metrics = {}
    for key in all_test_results[0].keys():
        avg_metrics[key] = np.mean([res[key] for res in all_test_results])

    # 4. 최종 평균값 TensorBoard 단일 기록
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f'logs_{config["dataset"]}/COMPARISON/DDPM_{args.prior_type}/step_{args.steps:03d}_{current_time}'
    summary_writer = tf.summary.create_file_writer(log_dir)
    
    with summary_writer.as_default():
        tf.summary.scalar('Test/Recall@10', avg_metrics['R@10'], step=args.steps)
        tf.summary.scalar('Test/Recall@20', avg_metrics['R@20'], step=args.steps)
        tf.summary.scalar('Test/NDCG@10', avg_metrics['N@10'], step=args.steps)
        tf.summary.scalar('Test/NDCG@20', avg_metrics['N@20'], step=args.steps)
        tf.summary.scalar('Test/Precision@10', avg_metrics['P@10'], step=args.steps)
        tf.summary.scalar('Test/Precision@20', avg_metrics['P@20'], step=args.steps)
        tf.summary.scalar('Test/Best_Step', args.steps, step=args.steps) # DDPM은 고정스텝 사용

    console.print(Panel.fit(
        f"🏆 [bold]FINAL AVERAGED TEST RESULT (Diffusion, {NUM_RUNS} Runs)[/] 🏆\n\n"
        f"Inference Steps: [bold cyan]{args.steps}[/] (Fixed)\n"
        f"K=10 | R: [red]{avg_metrics['R@10']:.4f}[/] | P: [green]{avg_metrics['P@10']:.4f}[/] | N: [blue]{avg_metrics['N@10']:.4f}[/] | H: [yellow]{avg_metrics['H@10']:.4f}[/]\n"
        f"K=20 | R: [red]{avg_metrics['R@20']:.4f}[/] | P: [green]{avg_metrics['P@20']:.4f}[/] | N: [blue]{avg_metrics['N@20']:.4f}[/] | H: [yellow]{avg_metrics['H@20']:.4f}[/]",
        border_style="cyan"
    ))

if __name__ == "__main__":
    train()