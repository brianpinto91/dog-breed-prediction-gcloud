docker_build_train_image:
	docker build -f Dockerfile_trainer -t ${IMAGE_URI} .
docker_run_local_train:
	docker run ${IMAGE_URI}
docker_push_to_gcloud:
	docker push ${IMAGE_URI}
make_bucket:
	gsutil mb -l ${REGION} gs://${BUCKET_NAME}
upload_data:
	gsutil cp data/* ${DATA_DIR}
gcloud_train:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--region ${REGION} \
  		--master-image-uri ${IMAGE_URI} \
		-- \
		--model_dir=${MODEL_DIR} \
		--epochs=3 \
		--use_cuda \
		--data_dir=${DATA_DIR} \
		--batch_size=60 \
		--test_batch_size=200 \
		--log_dir=${LOG_DIR} \