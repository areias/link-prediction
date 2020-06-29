#!/bin/sh


# zeppelin container

echo "Starting Zeppelin container..."

docker run -d \
	-p 8080:8080 \
	--rm \
	-v /home/areias/Documents/DataScience/graphs/logs:/logs \
	-v /home/areias/Documents/DataScience/graphs/notebook:/notebook \
	-v /home/areias/Documents/DataScience/graphs/data:/zeppelin/data \
	-e ZEPPELIN_LOG_DIR='/logs' \
	-e ZEPPELIN_NOTEBOOK_DIR='/notebook' \
	--name zeppelin \
	apache/zeppelin:0.9.0


# neo4j container
echo "Starting neo4j container..."

docker run -d \
	--publish=7473:7473 \
	--publish=7474:7474 \
	--publish=7687:7687 \
	--rm \
	--volume=/home/areias/Documents/DataScience/graphs/data:/data \
    --volume=/home/areias/Documents/DataScience/graphs/logs:/logs \
	--env NEO4J_AUTH=neo4j/test \
	--env NEO4JLABS_PLUGINS=["apoc","graphql"] \
	--name neo4j \
	neo4j

