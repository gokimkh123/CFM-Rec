# run_all.py
import os

# 실험할 스텝 리스트
step_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 200, 300, 400, 500]

print(f"🚀 [Pure Noise 실험] CFM-Rec 및 diffusion 실험을 시작합니다.")

for step in step_list:
    # --- 1. CFM-Rec (Flow) 실행 ---
    # --prior_type noise 인자를 추가합니다.
    print(f"\n[Flow - Pure Noise] Running with steps = {step} ...")
    flow_cmd = f"python train.py --steps {step} --prior_type noise"
    os.system(flow_cmd)

    # --- 2. diffusion (DDPM) 실행 ---
    # --prior_type noise 인자를 추가합니다.
    print(f"\n[Diffusion - Pure Noise] Running with steps = {step} ...")
    ddpm_cmd = f"python -m src_ddpm.train_ddpm --steps {step} --prior_type noise"
    os.system(ddpm_cmd)

print("\n✅ 모든 Pure Noise 실험이 완료되었습니다!")