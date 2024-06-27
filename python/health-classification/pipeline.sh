docker build -t rasheng/custom-model-health-classifier:v1 .
docker tag rasheng/custom-model-health-classifier:v1 asia-south1-docker.pkg.dev/fynd-cloud-non-prod/kserve/custom-model-health-classifier:v1
docker push asia-south1-docker.pkg.dev/fynd-cloud-non-prod/kserve/custom-model-health-classifier:v1
