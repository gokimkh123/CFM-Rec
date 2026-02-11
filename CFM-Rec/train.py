import tensorflow as tf
import yaml
import os
import glob
import numpy as np
import datetime
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn, TaskProgressColumn
)

from src.data_loader import ColdStartDataLoader
from src.model import FlowModel
from src.flow_logic import BernoulliFlow
from src.metrics import compute_metrics

console = Console()

parser = argparse.ArgumentParser()
parser.add_argument('--steps', type=int, default=10)
parser.add_argument('--prior_type', type=str, default='popularity')
args = parser.parse_args()

def load_config(path="config.yaml"):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

@tf.function
def train_step(model, optimizer, x_1, cond, t, x_0):
    mask = tf.cast(tf.random.uniform(tf.shape(x_1)) < t, tf.float32)
    x_t = mask * x_1 + (1.0 - mask) * x_0
    
    with tf.GradientTape() as tape:
        pred = model(x_t, cond, t, training=True)
        loss = tf.reduce_mean(tf.square(x_1 - pred))
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# ==============================================================================
# 1. Validation: 모든 스텝을 탐색(Search)하여 최적 스텝 도출
# 2. Test: 도출된 최적 스텝(fixed_step)으로 고정하여 평가
# ==============================================================================
def evaluate_user_to_item(model, flow, dataset, steps, k_list=[10, 20], fixed_step=None):
    """
    fixed_step=None  -> (Validation용) 1~steps 전체 탐색 후 Best Step 반환
    """
    
    # 모드에 따른 로그 출력
    if fixed_step is None:
        console.print(f"[bold cyan] [Validation] Searching Best Step across {steps} Steps...[/]")
        run_steps = steps
    else:
        console.print(f"[bold green] [Test] Running Inference with Fixed Best Step: {fixed_step}/{steps}[/]")
        run_steps = fixed_step

    # 결과 저장소
    step_outputs = {i: [] for i in range(1, run_steps + 1)} if fixed_step is None else {}
    test_outputs = [] # Test 모드일 때 마지막 결과만 저장
    all_targets = []
    
    for x_1, cond in dataset:
        batch_bs = tf.shape(x_1)[0]
        all_targets.append(x_1.numpy())
        
        curr_x = flow.get_prior_sample(batch_bs)
        dt = 1.0 / steps  # dt는 전체 steps(N=100) 기준으로 고정해야 궤적이 유지됨
        
        # 지정된 스텝만큼만 루프 실행
        for i in range(run_steps):
            t_val = i * dt
            t_tensor = tf.fill([batch_bs, 1], float(t_val))
            
            # 모델 예측
            pred = model(curr_x, cond, t_tensor, training=False)
            
            # 다음 상태로 이동
            curr_x = flow.inference_step(curr_x, pred, t_val, dt)
            
            # [저장 로직 분기]
            if fixed_step is None:
                # Validation: 모든 스텝 저장 (탐색용)
                step_outputs[i+1].append(pred.numpy())
            else:
                # Test: 마지막 스텝만 저장 (평가용)
                if i == run_steps - 1:
                    test_outputs.append(pred.numpy())

    # 정답 행렬 병합 및 전치
    target_matrix = np.concatenate(all_targets, axis=0)
    target_matrix_T = target_matrix.T
    num_users = target_matrix_T.shape[0]

    # --- [A] Validation 모드: Best Step 탐색 ---
    if fixed_step is None:
        best_step = -1
        best_recall = -1.0
        final_step_results = {}
        
        # 모든 스텝 평가
        for step in range(1, steps + 1):
            pred_matrix = np.concatenate(step_outputs[step], axis=0)
            pred_matrix_T = pred_matrix.T
            
            results = _calculate_metrics_batch(pred_matrix_T, target_matrix_T, num_users, k_list)
            final_step_results[step] = results
            
            if results['R@20'] > best_recall:
                best_recall = results['R@20']
                best_step = step
        
        best_result = final_step_results[best_step]
        best_result['Best_Step'] = best_step
        
        print(f"\n [Vali] Found Optimal Step: {best_step} (R@20: {best_recall:.4f})\n")
        return best_result

    # --- [B] Test 모드: 고정 스텝 평가 ---
    else:
        pred_matrix = np.concatenate(test_outputs, axis=0)
        pred_matrix_T = pred_matrix.T
        
        results = _calculate_metrics_batch(pred_matrix_T, target_matrix_T, num_users, k_list)
        results['Best_Step'] = fixed_step # 시각화를 위해 고정된 스텝 반환
        
        print(f"\n [Test] Final Result at Step {fixed_step}:")
        print(f"   R@20: {results['R@20']:.4f} | N@20: {results['N@20']:.4f}\n")
        return results

def _calculate_metrics_batch(pred_matrix_T, target_matrix_T, num_users, k_list):
    """메트릭 계산 보조 함수"""
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


def train():
    title = "Popularity Prior" if args.prior_type == 'popularity' else "Pure Noise Prior"
    console.print(Panel.fit(f"[bold yellow]CFM-Rec Training ({title}, N={args.steps})[/]", border_style="yellow"))
    
    # 모델 파일 정리
    for f in glob.glob("saved_model/best_flow_model*"):
        try: os.remove(f)
        except OSError: pass

    config = load_config()
    config['n_step'] = args.steps
    config['inference_steps'] = args.steps

    with console.status("[bold green]Loading Data...", spinner="dots"):
        loader = ColdStartDataLoader(config)
        num_items, num_users = loader.build()
        train_ds = loader.get_dataset(mode='train')
        vali_ds = loader.get_dataset(mode='vali')
        test_ds = loader.get_dataset(mode='test')
        
        user_activity = tf.convert_to_tensor(loader.user_activity, dtype=tf.float32)
        flow = BernoulliFlow(loader.user_activity, prior_type=args.prior_type)

    model_dims = config['dims_mlp'] + [num_users]
    model = FlowModel(model_dims, config['time_embedding_size'], config.get('dropout', 0.0))
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
    
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f'logs_{config["dataset"]}/COMPARISON/FLOW_{args.prior_type}/step_{args.steps:03d}_{current_time}'
    summary_writer = tf.summary.create_file_writer(log_dir)
    
    epochs = config['epochs']
    eval_step = config.get('eval_step', 10)
    
    best_recall = -1.0
    best_val_step = args.steps # 기본값 (Validation 전까지 사용)
    patience_cnt = 0
    
    steps_per_epoch = int(np.ceil(loader.num_entities / config['batch_size']))
    progress = Progress(
        SpinnerColumn(), TextColumn("[bold blue]{task.description}"), BarColumn(),
        TaskProgressColumn(), TimeRemainingColumn(), TextColumn("{task.fields[info]}"), console=console
    )

    with progress:
        overall_task = progress.add_task("[bold magenta]Total Progress", total=epochs, info="")
        epoch_task = progress.add_task("[cyan]Current Epoch", total=steps_per_epoch, info="Loss: N/A")

        for epoch in range(epochs):
            progress.reset(epoch_task)
            progress.update(epoch_task, description=f"[cyan]Epoch {epoch+1}/{epochs}")
            
            # --- Train Phase ---
            train_loss, train_steps = 0, 0
            for x_1, cond in train_ds:
                curr_bs = tf.shape(x_1)[0]
                t = tf.cast(tf.random.uniform((curr_bs, 1), 1, args.steps+1, dtype=tf.int32), tf.float32) / args.steps
                x_0 = flow.get_prior_sample(curr_bs)
                loss = train_step(model, optimizer, x_1, cond, t, x_0)
                train_loss += loss.numpy()
                train_steps += 1
                progress.update(epoch_task, advance=1, info=f"Loss: {loss.numpy():.4f}")
            
            avg_loss = train_loss / train_steps
            with summary_writer.as_default():
                tf.summary.scalar('Loss/train', avg_loss, step=epoch)
            
            # --- Validation Phase ---
            if (epoch + 1) % eval_step == 0:
                progress.update(epoch_task, description="[bold yellow]Validating (Search Best Step)...", info="")
                
                # [Validation] fixed_step=None으로 호출하여 최적 스텝 탐색
                val_metrics = evaluate_user_to_item(model, flow, vali_ds, args.steps, k_list=[10, 20], fixed_step=None)
                
                r10, r20 = val_metrics['R@10'], val_metrics['R@20']
                
                with summary_writer.as_default():
                    tf.summary.scalar('Metrics/Recall@10', r10, step=epoch)
                    tf.summary.scalar('Metrics/Recall@20', r20, step=epoch)

                log_msg = f"E{epoch+1:03d} | Loss: {avg_loss:.4f} | Val R@10: {r10:.4f} | Val R@20: {r20:.4f}"
                
                if r20 > best_recall:
                    best_recall = r20
           
                    best_val_step = val_metrics['Best_Step']
                    patience_cnt = 0
                    model.save_weights("saved_model/best_flow_model")
                    log_msg += f" [bold green]★ Best (Step {best_val_step})[/]"
                else:
                    patience_cnt += 1
                
                console.print(log_msg)
                if patience_cnt >= config.get('patience', 10): break
            progress.update(overall_task, advance=1)

    # --- Final Test Phase ---
    console.print(f"\n[bold yellow] Final Test with Optimal Step found in Validation: {best_val_step}[/]")
    try: model.load_weights("saved_model/best_flow_model")
    except: pass

    # [Test] 저장해둔 best_val_step을 고정값으로 전달
    test_metrics = evaluate_user_to_item(model, flow, test_ds, args.steps, k_list=[10, 20], fixed_step=best_val_step)
    
    final_r10, final_r20 = test_metrics['R@10'], test_metrics['R@20']
    final_n10, final_n20 = test_metrics['N@10'], test_metrics['N@20']
    final_p10, final_p20 = test_metrics['P@10'], test_metrics['P@20']
    
    with summary_writer.as_default():
        tf.summary.scalar('Test/Recall@10', final_r10, step=epochs)
        tf.summary.scalar('Test/Recall@20', final_r20, step=epochs)
        tf.summary.scalar('Test/NDCG@10', final_n10, step=epochs)
        tf.summary.scalar('Test/NDCG@20', final_n20, step=epochs)
        tf.summary.scalar('Test/Precision@10', final_p10, step=epochs)
        tf.summary.scalar('Test/Precision@20', final_p20, step=epochs)
        tf.summary.scalar('Test/Best_Step', best_val_step, step=epochs)

    console.print(Panel.fit(
        f" [bold]FINAL TEST RESULT (CFM-Rec)[/] \n\n"
        f"Used Inference Step: [bold cyan]{best_val_step}[/] (Fixed from Vali)\n"
        f"K=10 | R: [red]{test_metrics['R@10']:.4f}[/] | P: [green]{test_metrics['P@10']:.4f}[/] | N: [blue]{test_metrics['N@10']:.4f}[/] | H: [yellow]{test_metrics['H@10']:.4f}[/]\n"
        f"K=20 | R: [red]{test_metrics['R@20']:.4f}[/] | P: [green]{test_metrics['P@20']:.4f}[/] | N: [blue]{test_metrics['N@20']:.4f}[/] | H: [yellow]{test_metrics['H@20']:.4f}[/]",
        border_style="magenta"
    ))

if __name__ == "__main__":
    if not os.path.exists("saved_model"): os.makedirs("saved_model")
    train()