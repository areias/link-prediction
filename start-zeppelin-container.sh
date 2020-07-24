#!/bin/sh

set -x

echo "Starting Zeppelin container..."

docker run -d \
	-u `id -u $USER` \
	-p 8080:8080 \
	--rm \
	-v /home/areias/Documents/DataScience/graphs/logs:/logs \
	-v /home/areias/Documents/DataScience/graphs/notebook:/notebook \
	-v /home/areias/Documents/DataScience/graphs/jars:/jars \
	-v /home/areias/Documents/DataScience/graphs/data:/zeppelin/data \
	-e ZEPPELIN_LOG_DIR='/logs' \
	-e ZEPPELIN_NOTEBOOK_DIR='/notebook' \
	--name zeppelin \
	apache/zeppelin:0.9.0
