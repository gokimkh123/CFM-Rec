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
# [핵심] evaluate.py와 동일한 방식의 User-to-Item 평가 함수
# ==============================================================================
def evaluate_user_to_item(model, flow, dataset, steps, k_list=[10, 20]):
    """
    모든 Cold Item에 대한 예측을 수행한 뒤,
    행렬을 전치(Transpose)하여 [Users x Items] 관점에서 평가합니다.
    """
    all_preds = []
    all_targets = []
    
    # 1. 배치 단위로 추론 (Item-based)
    for x_1, cond in dataset:
        batch_bs = tf.shape(x_1)[0]
        curr_x = flow.get_prior_sample(batch_bs)
        dt = 1.0 / steps
        
        for i in range(steps):
            t_val = i * dt
            t_tensor = tf.fill([batch_bs, 1], float(t_val))
            pred = model(curr_x, cond, t_tensor, training=False)
            curr_x = flow.inference_step(curr_x, pred, t_val, dt)
            
        all_preds.append(curr_x.numpy())
        all_targets.append(x_1.numpy())
        
    # 2. 전체 행렬 병합 (Items x Users)
    pred_matrix = np.concatenate(all_preds, axis=0)
    target_matrix = np.concatenate(all_targets, axis=0)
    
    # 3. User 관점으로 전치 (Users x Items)
    # 이제 row는 User가 되고, col은 Test set의 Cold Items가 됩니다.
    pred_matrix_T = pred_matrix.T
    target_matrix_T = target_matrix.T
    
    num_users = pred_matrix_T.shape[0]
    results = {f'R@{k}': [] for k in k_list}
    results.update({f'N@{k}': [] for k in k_list})

    # 4. 각 유저별로 평가
    for u in range(num_users):
        # 정답: 이 유저가 좋아한 Cold Items (Test set 내에서)
        gt_items = np.where(target_matrix_T[u] > 0.5)[0]
        if len(gt_items) == 0: continue 
        
        # 예측: 점수가 높은 순서대로 아이템 인덱스 추출
        top_indices = np.argsort(pred_matrix_T[u])[-max(k_list):][::-1]
        
        m = compute_metrics(top_indices, gt_items, k_list=k_list)
        for k in k_list:
            results[f'R@{k}'].append(m[f'Recall@{k}'])
            results[f'N@{k}'].append(m[f'NDCG@{k}'])
            
    final_metrics = {}
    for k in k_list:
        final_metrics[f'R@{k}'] = np.mean(results[f'R@{k}']) if results[f'R@{k}'] else 0.0
        final_metrics[f'N@{k}'] = np.mean(results[f'N@{k}']) if results[f'N@{k}'] else 0.0
        
    return final_metrics

def train():
    title = "Popularity Prior" if args.prior_type == 'popularity' else "Pure Noise Prior"
    console.print(Panel.fit(f"[bold yellow]CFM-Rec Training ({title}, N={args.steps})[/]", border_style="yellow"))
    # 기존 모델 파일 정리
    for f in glob.glob("saved_model/best_flow_model*"):
        try: os.remove(f)
        except OSError: pass

    config = load_config()
    config['n_step'] = args.steps
    config['inference_steps'] = args.steps

    with console.status("[bold green]Loading Data...", spinner="dots"):
        loader = ColdStartDataLoader(config)
        
        # [수정 완료] build()의 반환값을 언패킹하여 받습니다.
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
    log_dir = f'logs/COMPARISON/FLOW_{args.prior_type}/step_{args.steps:03d}_{current_time}'
    summary_writer = tf.summary.create_file_writer(log_dir)
    epochs = config['epochs']
    eval_step = config.get('eval_step', 10)
    best_recall = -1.0
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
                probs = tf.tile(tf.expand_dims(user_activity, 0), [curr_bs, 1])
                x_0 = tf.cast(tf.random.uniform(tf.shape(x_1)) < probs, tf.float32)
                
                loss = train_step(model, optimizer, x_1, cond, t, x_0)
                train_loss += loss.numpy()
                train_steps += 1
                progress.update(epoch_task, advance=1, info=f"Loss: {loss.numpy():.4f}")
            
            avg_loss = train_loss / train_steps
            with summary_writer.as_default():
                tf.summary.scalar('Loss/train', avg_loss, step=epoch)
            
            # --- Validation Phase (User-to-Item) ---
            if (epoch + 1) % eval_step == 0:
                progress.update(epoch_task, description="[bold yellow]Validating (User-to-Item)...", info="")
                
                val_metrics = evaluate_user_to_item(model, flow, vali_ds, args.steps, k_list=[10, 20])
                r10, r20 = val_metrics['R@10'], val_metrics['R@20']

                with summary_writer.as_default():
                    tf.summary.scalar('Metrics/Recall@10', r10, step=epoch)
                    tf.summary.scalar('Metrics/Recall@20', r20, step=epoch)

                log_msg = f"E{epoch+1:03d} | Loss: {avg_loss:.4f} | Val R@10: {r10:.4f} | Val R@20: {r20:.4f}"
                if r20 > best_recall:
                    best_recall = r20
                    patience_cnt = 0
                    model.save_weights("saved_model/best_flow_model")
                    log_msg += " [bold green]★ Best[/]"
                else:
                    patience_cnt += 1
                
                console.print(log_msg)
                if patience_cnt >= config.get('patience', 10): break
            progress.update(overall_task, advance=1)

    # --- Final Test Phase (User-to-Item) ---
    console.print("\n[bold yellow]🚀 Running Final User-to-Item Evaluation on TEST SET...[/]")
    try: model.load_weights("saved_model/best_flow_model")
    except: pass

    test_metrics = evaluate_user_to_item(model, flow, test_ds, args.steps, k_list=[10, 20])
    
    final_r10, final_r20 = test_metrics['R@10'], test_metrics['R@20']
    final_n20 = test_metrics['N@20']

    with summary_writer.as_default():
        tf.summary.scalar('Test/Recall@10', final_r10, step=epochs)
        tf.summary.scalar('Test/Recall@20', final_r20, step=epochs)
        tf.summary.scalar('Test/NDCG@20', final_n20, step=epochs)

    console.print(Panel.fit(
        f"🏆 FINAL TEST RESULT (User-to-Item) 🏆\n\n"
        f"Recall@10 : [bold red]{final_r10:.4f}[/]\n"
        f"Recall@20 : [bold red]{final_r20:.4f}[/]\n"
        f"NDCG@20   : [bold red]{final_n20:.4f}[/]",
        border_style="red"
    ))

if __name__ == "__main__":
    if not os.path.exists("saved_model"): os.makedirs("saved_model")
    train()