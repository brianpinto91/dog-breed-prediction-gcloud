docker_build_train_image_cpu:
	docker build -f Dockerfile_trainer_cpu -t ${IMAGE_URI} .
docker_build_train_image_gpu:
	docker build -f Dockerfile_trainer_gpu -t ${IMAGE_URI} .
docker_run_local_train:
	docker run ${IMAGE_URI}
docker_push_to_gcloud:
	docker push ${IMAGE_URI}
make_bucket:
	gsutil mb -l ${REGION} gs://${BUCKET_NAME}
upload_data:
	gsutil cp data/* ${DATA_DIR}
gcloud_train_cpu:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--region ${REGION} \
  		--master-image-uri ${IMAGE_URI} \
		-- \
		--model_dir=${MODELS_DIR} \
		--epochs=1 \
		--use_cuda \
		--data_dir=${DATA_DIR} \
		--batch_size=60 \
		--test_batch_size=200 \
		--log_dir=${LOGS_DIR}
gcloud_train_gpu:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--region ${REGION} \
		--scale-tier BASIC_GPU \
  		--master-image-uri ${IMAGE_URI} \
		-- \
		--model_dir=${MODELS_DIR} \
		--epochs=10 \
		--use_cuda \
		--data_dir=${DATA_DIR} \
		--batch_size=60 \
		--test_batch_size=300 \
		--log_dir=${LOGS_DIR}