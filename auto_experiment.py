import subprocess
import yaml
import os
import re
import time
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# 1. 설정 및 유틸리티
# =========================================================
# 실험할 N 스텝 리스트
COMPARISON_STEPS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 50, 100, 200, 300]  # 필요에 따라 수정하세요
COMMON_EPOCHS = 100
STOPPING_STEP = 15  # Early Stopping Patience

results = {
    'diff_r20': [], 'diff_time': [],
    'flow_best_r20': [], 'flow_best_s': [], 'flow_best_time': [],
    'best_model_n': 0, 'best_model_path': "" 
}

def run_command(cmd, capture=True):
    if capture:
        res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if res.returncode != 0:
            return None, res.stdout
        return res.stdout, None
    else:
        res = subprocess.run(cmd, shell=True)
        if res.returncode != 0:
            return None, "Process Failed"
        return "", None

def update_yaml(file_path, **kwargs):
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    data.update(kwargs)
    with open(file_path, 'w') as f:
        yaml.dump(data, f)

def get_latest_checkpoint(dir="saved/"):
    if not os.path.exists(dir): return None
    files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.pth')]
    return max(files, key=os.path.getctime) if files else None

def parse_result(output):
    if not output: return 0.0, 0.0
    
    # [방법 1] 확실한 토큰(치트키) 찾기
    token_match = re.search(r"__PARSE_RESULT__:([\d.,eE\-]+)", output)
    if token_match:
        try:
            content = token_match.group(1)
            parts = content.split(',')
            return float(parts[1]), float(parts[2])
        except (ValueError, IndexError):
            pass

    # [방법 2] 토큰이 없을 경우 일반 텍스트 파싱
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    clean_output = ansi_escape.sub('', output)
    
    r20 = re.search(r"recall@20\s*[:=]\s*([\d.]+)", clean_output, re.IGNORECASE)
    t_test = re.search(r"test result.*?(\d+\.\d+)s", clean_output, re.S | re.IGNORECASE)
    
    if not r20:
        r20 = re.search(r"['\"]recall@20['\"]\s*[:=]\s*([\d.]+)", clean_output, re.IGNORECASE)

    r_val = float(r20.group(1)) if r20 else 0.0
    t_val = float(t_test.group(1)) if t_test else 0.0
    
    return r_val, t_val

# =========================================================
# 2. [실험 A] N 변화에 따른 성능 및 속도 비교
# =========================================================
print("\n" + "="*60)
print(" PHASE A: Performance & Efficiency Comparison")
print(f" Target Steps: {COMPARISON_STEPS}")
print("="*60)

global_best_r20 = -1.0

for i, n in enumerate(COMPARISON_STEPS):
    print(f"\n>>> [A-{i+1}/{len(COMPARISON_STEPS)}] Experimenting at N = {n}")
    
    # --- [A-1] DiffCF ---
    print(f"   [DiffCF] Training...")
    update_yaml('diffcf.yaml', n_steps=n, s_steps=None, epochs=COMMON_EPOCHS, stopping_step=STOPPING_STEP)
    
    out, _ = run_command("python run.py --config diffcf.yaml", capture=False)
    
    diff_val = 0.0
    diff_t = 0.0
    if out is not None:
        ckpt = get_latest_checkpoint()
        out_eval, _ = run_command(f"python evaluate.py --config diffcf.yaml --checkpoint {ckpt}", capture=True)
        diff_val, diff_t = parse_result(out_eval)
        print(f"      -> DiffCF Result: R@20={diff_val:.4f}, Time={diff_t:.2f}s")
    
    results['diff_r20'].append(diff_val)
    results['diff_time'].append(diff_t)

    # --- [A-2] FlowCF ---
    print(f"   [FlowCF] Training...")
    update_yaml('flowcf.yaml', n_steps=n, s_steps=1, epochs=COMMON_EPOCHS, stopping_step=STOPPING_STEP)
    
    out, _ = run_command("python run.py --config flowcf.yaml --act leakyrelu", capture=False)
    
    flow_best_val = 0.0
    flow_best_s = 1
    flow_best_t = 0.0
    
    if out is not None:
        ckpt_flow = get_latest_checkpoint()
        
        # 탐색 전략
        if n <= 20:
            s_candidates = list(range(1, n + 1))
        else:
            base = list(range(1, 11))
            stride = max(10, n // 10)
            sparse = list(range(stride, n, stride))
            s_candidates = sorted(list(set(base + sparse)))
            if s_candidates[-1] != n: s_candidates.append(n)
            
        print(f"      -> Searching Best S in: {s_candidates}")
        
        for s in s_candidates:
            update_yaml('flowcf.yaml', n_steps=n, s_steps=s)
            out_eval, _ = run_command(f"python evaluate.py --config flowcf.yaml --checkpoint {ckpt_flow}", capture=True)
            curr_r20, curr_t = parse_result(out_eval)
            
            if curr_r20 > flow_best_val:
                flow_best_val = curr_r20
                flow_best_s = s
                flow_best_t = curr_t
            elif curr_r20 == flow_best_val and curr_r20 > 0:
                if curr_t < flow_best_t:
                    flow_best_t = curr_t
                    flow_best_s = s
        
        print(f"      -> FlowCF Best: R@20={flow_best_val:.4f} (at S={flow_best_s}, Time={flow_best_t:.2f}s)")
        
        if flow_best_val > global_best_r20:
            global_best_r20 = flow_best_val
            results['best_model_n'] = n
            results['best_model_path'] = ckpt_flow
            print(f"      *** New Best Model Found! (N={n}) ***")

    results['flow_best_r20'].append(flow_best_val)
    results['flow_best_s'].append(flow_best_s)
    results['flow_best_time'].append(flow_best_t)

# --- 그래프 설정 ---
try: plt.style.use('seaborn-v0_8-darkgrid')
except: plt.style.use('ggplot')

# [Graph 1] Performance Comparison
plt.figure(figsize=(10, 6))
plt.plot(COMPARISON_STEPS, results['diff_r20'], 'o--', label='DiffCF', color='blue', alpha=0.7)
plt.plot(COMPARISON_STEPS, results['flow_best_r20'], 's-', label='FlowCF (Best S)', color='red', linewidth=2)

plt.xticks(COMPARISON_STEPS, COMPARISON_STEPS) 

# [수정됨] DiffCF 최고 성능 X 표시 추가
if len(results['diff_r20']) > 0:
    max_y_diff = max(results['diff_r20'])
    max_idx_diff = results['diff_r20'].index(max_y_diff)
    max_x_diff = COMPARISON_STEPS[max_idx_diff]
    # 파란색 X 표시
    plt.scatter(max_x_diff, max_y_diff, marker='x', s=200, color='blue', linewidth=3, zorder=10, label=f'Peak DiffCF (N={max_x_diff})')

# [수정됨] FlowCF 최고 성능 X 표시
if len(results['flow_best_r20']) > 0:
    max_y_flow = max(results['flow_best_r20'])
    max_idx_flow = results['flow_best_r20'].index(max_y_flow)
    max_x_flow = COMPARISON_STEPS[max_idx_flow]
    # 빨간색 X 표시 (검은색보다 구분하기 쉽게 색상 매칭)
    plt.scatter(max_x_flow, max_y_flow, marker='x', s=200, color='red', linewidth=3, zorder=10, label=f'Peak FlowCF (N={max_x_flow})')

plt.xlabel('Training Steps (N)') 
plt.ylabel('Recall@20')
plt.title('Experiment A: Performance Comparison')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.savefig('result_A_performance.png')
print("\n[Saved] result_A_performance.png")

# [Graph 2] Inference Time Comparison
plt.figure(figsize=(10, 6))
plt.plot(COMPARISON_STEPS, results['diff_time'], 'o--', label='DiffCF', color='blue', alpha=0.7)
plt.plot(COMPARISON_STEPS, results['flow_best_time'], 's-', label='FlowCF (Best S)', color='green', linewidth=2)

plt.xticks(COMPARISON_STEPS, COMPARISON_STEPS)

plt.xlabel('Training Steps (N)')
plt.ylabel('Inference Time (seconds)')
plt.title('Experiment A: Inference Speed Comparison')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.savefig('result_A_inference_time.png')
print("[Saved] result_A_inference_time.png")


# =========================================================
# 3. [실험 B] 최고 모델의 S-Step 정밀 분석
# =========================================================
best_n = results['best_model_n']
best_ckpt = results['best_model_path']

if best_n > 0 and os.path.exists(best_ckpt):
    print("\n" + "="*60)
    print(f" PHASE B: Efficiency Analysis on Best Model (N={best_n})")
    print(f" Checkpoint: {best_ckpt}")
    print("="*60)
    
    b_s_steps = []
    b_r20s = []
    
    print(f"   -> Scanning ALL steps from 1 to {best_n}...")
    full_scan_steps = list(range(1, best_n + 1))
    
    for idx, s in enumerate(full_scan_steps):
        if idx % 10 == 0: print(f"      Processing S={s}...", end="\r")
        update_yaml('flowcf.yaml', n_steps=best_n, s_steps=s)
        out_eval, _ = run_command(f"python evaluate.py --config flowcf.yaml --checkpoint {best_ckpt}", capture=True)
        r20, _ = parse_result(out_eval) 
        
        b_s_steps.append(s)
        b_r20s.append(r20)
    
    print(f"      Done.                                ")

    plt.figure(figsize=(10, 6))
    plt.plot(b_s_steps, b_r20s, '-', color='green', linewidth=2)
    plt.scatter(b_s_steps, b_r20s, color='green', s=20, alpha=0.6)
    
    max_y = max(b_r20s)
    max_x = b_s_steps[b_r20s.index(max_y)]
    plt.plot(max_x, max_y, 'r*', markersize=15, label=f'Peak (S={max_x})')
    
    plt.xlabel(f'Inference Steps (S) [1 ~ {best_n}]')
    plt.ylabel('Recall@20')
    plt.title(f'Experiment B: Efficiency Trade-off (Best Model N={best_n})')
    plt.legend()
    plt.grid(True)
    plt.savefig('result_B_efficiency.png')
    print("\n[Saved] result_B_efficiency.png")

else:
    print("\n[Skip] Experiment B skipped.")

print("\nAll Experiments Completed.")