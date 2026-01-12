# Dockerfile

# 1. Base Image (PyTorch + CUDA)
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# 2. 기본 설정
WORKDIR /app
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 3. 필수 시스템 도구 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 4. LightGBM, XGBoost (Conda로 설치하여 의존성 문제 방지)
RUN conda install -y lightgbm xgboost && conda clean -ya

# 5. Python 라이브러리 설치
# RecBole과 호환성을 위해 numpy 버전을 2.0 미만으로 고정
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "numpy<2.0" recbole pyyaml

# 6. 소스 코드 복사 (현재 폴더의 모든 파일을 컨테이너의 /app으로 복사)
COPY . /app

# 7. [핵심] 커스텀 utils.py 패치
# 우리가 수정한(Pandas 호환성 해결된) utils.py를 RecBole 설치 경로에 덮어씌웁니다.
RUN python -c "import recbole.data; import os; import shutil; \
    dest = os.path.join(os.path.dirname(recbole.data.__file__), 'utils.py'); \
    shutil.copyfile('/app/utils.py', dest); \
    print(f'Successfully injected custom utils.py to: {dest}')"

# 8. 기본 실행 명령어 (학습)
CMD ["python", "run.py", "--config", "flowcf.yaml"]