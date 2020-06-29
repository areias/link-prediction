#!/bin/sh
set -x #echo on

docker network create --driver bridge mynetwork


# neo4j container
echo "Starting neo4j container..."

docker run -d \
	-u `id -u $USER` \
	--publish=7473:7473 \
	--publish=7474:7474 \
	--publish=7687:7687 \
	--rm \
	--volume=/home/areias/Documents/DataScience/graphs/data:/data \
    --volume=/home/areias/Documents/DataScience/graphs/logs:/logs \
	--env NEO4J_AUTH=neo4j/test \
	--env NEO4JLABS_PLUGINS=["apoc","graphql"] \
	--env NEO4J_dbms.connectors.default_listen_address=0.0.0.0 \
	--name neo4j \
	neo4j

sleep 5

set -x

# jupyter container
echo "Starting jupyter container..."
docker run -d \
	--rm \
	-p 8888:8888 \
	-v /home/areias/Documents/DataScience/graphs:/home/jovyan \
	--env JUPYTER_TOKEN='easy' \
	--name jupyter \
	jupyter/datascience-notebook

