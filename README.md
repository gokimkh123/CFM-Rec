# Cold_Item_FlowCF
Use flowCF for item cold start situations


# docker build 
docker compose build

# Training
docker compose up

# Inference
docker compose run --rm flowcf python inference.py --checkpoint saved/FlowCF-Jan-12-2026_00-00-00.pth
