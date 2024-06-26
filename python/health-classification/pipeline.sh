docker build -t rasheng/default-custom-model-mnist-tf:v1 .
docker tag rasheng/default-custom-model-mnist-tf:v1 asia-south1-docker.pkg.dev/fynd-cloud-non-prod/kserve/default-custom-model-mnist-tf:v1
docker push asia-south1-docker.pkg.dev/fynd-cloud-non-prod/kserve/default-custom-model-mnist-tf:v1
