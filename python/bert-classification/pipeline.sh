docker build -t rasheng/custom-model-bert-classification:v1 .
docker tag rasheng/custom-model-bert-classification:v1 asia-south1-docker.pkg.dev/fynd-cloud-non-prod/kserve/custom-model-bert-classification:v1
docker push asia-south1-docker.pkg.dev/fynd-cloud-non-prod/kserve/custom-model-bert-classification:v1
