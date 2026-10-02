# Shri Krishna: Sharanam Mam Jai Ambe Jai Amumaiya Jai Devudada Shri Ganeshay Nama: Shri Saraswatiyay Nama:
#!/bin/bash -e
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 303666750405.dkr.ecr.us-west-2.amazonaws.com
docker build -t 303666750405.dkr.ecr.us-west-2.amazonaws.com/risk-prediction/mirai:XAI . --platform linux/amd64
docker push 303666750405.dkr.ecr.us-west-2.amazonaws.com/risk-prediction/mirai:XAI