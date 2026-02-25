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

def evaluate_user_to_item(model, flow, dataset, steps, k_list=[10, 20], fixed_step=None):
    if fixed_step is None:
        run_steps = steps
    else:
        run_steps = fixed_step

    step_outputs = {i: [] for i in range(1, run_steps + 1)} if fixed_step is None else {}
    test_outputs = [] 
    all_targets = []
    
    for x_1, cond in dataset:
        batch_bs = tf.shape(x_1)[0]
        all_targets.append(x_1.numpy())
        
        curr_x = flow.get_prior_sample(batch_bs)
        dt = 1.0 / steps  
        
        for i in range(run_steps):
            t_val = i * dt
            t_tensor = tf.fill([batch_bs, 1], float(t_val))
            
            pred = model(curr_x, cond, t_tensor, training=False)
            curr_x = flow.inference_step(curr_x, pred, t_val, dt)
            
            if fixed_step is None:
                step_outputs[i+1].append(pred.numpy())
            else:
                if i == run_steps - 1:
                    test_outputs.append(pred.numpy())

    target_matrix = np.concatenate(all_targets, axis=0)
    target_matrix_T = target_matrix.T
    num_users = target_matrix_T.shape[0]

    if fixed_step is None:
        best_step = -1
        best_recall = -1.0
        final_step_results = {}
        
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
        return best_result

    else:
        pred_matrix = np.concatenate(test_outputs, axis=0)
        pred_matrix_T = pred_matrix.T
        
        results = _calculate_metrics_batch(pred_matrix_T, target_matrix_T, num_users, k_list)
        results['Best_Step'] = fixed_step 
        return results

def _calculate_metrics_batch(pred_matrix_T, target_matrix_T, num_users, k_list):
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
        flow = BernoulliFlow(loader.user_activity, prior_type=args.prior_type)

    epochs = config['epochs']
    eval_step = config.get('eval_step', 10)
    steps_per_epoch = int(np.ceil(loader.num_entities / config['batch_size']))
    
    NUM_RUNS = 5
    all_test_results = []
    random.seed(2026)
    for run in range(NUM_RUNS):
        sampled_lr = random.choice([0.0005, 0.001])
        sampled_time_emb = random.choice([32, 64])
        sampled_dropout = random.choice([0.0, 0.1, 0.2])
        
        console.print(f"\n[bold magenta]=== Run {run+1}/{NUM_RUNS} | LR: {sampled_lr} | TimeEmb: {sampled_time_emb} | Dropout: {sampled_dropout} ===[/]")

        model_dims = config['dims_mlp'] + [num_users]
        model = FlowModel(model_dims, sampled_time_emb, sampled_dropout)
        optimizer = tf.keras.optimizers.Adam(learning_rate=sampled_lr)

        # 모델 및 옵티마이저 초기화 이후 내부 스코프에 정적 연산 그래프 정의
        @tf.function
        def step_fn(x_1, cond, t, x_0):
            mask = tf.cast(tf.random.uniform(tf.shape(x_1)) < t, tf.float32)
            x_t = mask * x_1 + (1.0 - mask) * x_0
            
            with tf.GradientTape() as tape:
                pred = model(x_t, cond, t, training=True)
                loss = tf.reduce_mean(tf.square(x_1 - pred))
            
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return loss
        
        best_recall = -1.0
        best_val_step = args.steps
        patience_cnt = 0
        current_save_path = f"saved_model/best_flow_model_run{run}"

        progress = Progress(
            SpinnerColumn(), TextColumn("[bold blue]{task.description}"), BarColumn(),
            TaskProgressColumn(), TimeRemainingColumn(), TextColumn("{task.fields[info]}"), console=console
        )

        with progress:
            overall_task = progress.add_task(f"[bold magenta]Run {run+1} Progress", total=epochs, info="")
            epoch_task = progress.add_task("[cyan]Current Epoch", total=steps_per_epoch, info="Loss: N/A")

            for epoch in range(epochs):
                progress.reset(epoch_task)
                progress.update(epoch_task, description=f"[cyan]Epoch {epoch+1}/{epochs}")
                
                train_loss, train_steps = 0, 0
                for x_1, cond in train_ds:
                    curr_bs = tf.shape(x_1)[0]
                    t = tf.cast(tf.random.uniform((curr_bs, 1), 1, args.steps+1, dtype=tf.int32), tf.float32) / args.steps
                    x_0 = flow.get_prior_sample(curr_bs)
                    
                    # 내부 step_fn 호출
                    loss = step_fn(x_1, cond, t, x_0)
                    
                    train_loss += loss.numpy()
                    train_steps += 1
                    progress.update(epoch_task, advance=1, info=f"Loss: {loss.numpy():.4f}")
                
                avg_loss = train_loss / train_steps
                
                if (epoch + 1) % eval_step == 0:
                    val_metrics = evaluate_user_to_item(model, flow, vali_ds, args.steps, k_list=[10, 20], fixed_step=None)
                    r10, r20 = val_metrics['R@10'], val_metrics['R@20']
                    
                    log_msg = f"E{epoch+1:03d} | Loss: {avg_loss:.4f} | Val R@10: {r10:.4f} | Val R@20: {r20:.4f}"
                    
                    if r20 > best_recall:
                        best_recall = r20
                        best_val_step = val_metrics['Best_Step']
                        patience_cnt = 0
                        model.save_weights(current_save_path)
                        log_msg += f" [bold green]★ Best (Step {best_val_step})[/]"
                    else:
                        patience_cnt += 1
                    
                    console.print(log_msg)
                    if patience_cnt >= config.get('patience', 10): 
                        console.print("[bold red]Early Stopping Triggered.[/]")
                        break
                progress.update(overall_task, advance=1)

        try: model.load_weights(current_save_path)
        except: pass
        
        test_metrics = evaluate_user_to_item(model, flow, test_ds, args.steps, k_list=[10, 20], fixed_step=best_val_step)
        all_test_results.append(test_metrics)
        
        # 메모리 반환
        tf.keras.backend.clear_session()

    console.print(f"\n[bold yellow]=== Calculating Average over {NUM_RUNS} runs ===[/]")
    avg_metrics = {}
    for key in all_test_results[0].keys():
        avg_metrics[key] = np.mean([res[key] for res in all_test_results])

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f'logs_{config["dataset"]}/COMPARISON/FLOW_{args.prior_type}/step_{args.steps:03d}_{current_time}'
    summary_writer = tf.summary.create_file_writer(log_dir)
    
    with summary_writer.as_default():
        tf.summary.scalar('Test/Recall@10', avg_metrics['R@10'], step=args.steps)
        tf.summary.scalar('Test/Recall@20', avg_metrics['R@20'], step=args.steps)
        tf.summary.scalar('Test/NDCG@10', avg_metrics['N@10'], step=args.steps)
        tf.summary.scalar('Test/NDCG@20', avg_metrics['N@20'], step=args.steps)
        tf.summary.scalar('Test/Precision@10', avg_metrics['P@10'], step=args.steps)
        tf.summary.scalar('Test/Precision@20', avg_metrics['P@20'], step=args.steps)
        tf.summary.scalar('Test/Best_Step', avg_metrics['Best_Step'], step=args.steps)

    console.print(Panel.fit(
        f" [bold]FINAL AVERAGED TEST RESULT ({NUM_RUNS} Runs)[/] \n\n"
        f"Avg Optimal Step: [bold cyan]{avg_metrics['Best_Step']:.1f}[/]\n"
        f"K=10 | R: [red]{avg_metrics['R@10']:.4f}[/] | P: [green]{avg_metrics['P@10']:.4f}[/] | N: [blue]{avg_metrics['N@10']:.4f}[/] | H: [yellow]{avg_metrics['H@10']:.4f}[/]\n"
        f"K=20 | R: [red]{avg_metrics['R@20']:.4f}[/] | P: [green]{avg_metrics['P@20']:.4f}[/] | N: [blue]{avg_metrics['N@20']:.4f}[/] | H: [yellow]{avg_metrics['H@20']:.4f}[/]",
        border_style="magenta"
    ))

if __name__ == "__main__":
    if not os.path.exists("saved_model"): os.makedirs("saved_model")
    train()