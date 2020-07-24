#!/bin/sh


# neo4j container
echo "Starting neo4j container..."

set -x #echo on

docker run \
	-u `id -u $USER` \
	--publish=7473:7473 \
	--publish=7474:7474 \
	--publish=7687:7687 \
	--rm \
	--volume=/home/areias/Documents/DataScience/graphs/data:/data \
	--volume=/home/areias/Documents/DataScience/graphs/logs:/logs \
        --volume=/home/areias/Documents/DataScience/graphs/plugins:/plugins \
        --volume=/home/areias/Documents/DataScience/graphs/import:/import \
	--env NEO4J_AUTH=neo4j/test \
    -e NEO4J_apoc_export_file_enabled=true \
    -e NEO4J_apoc_import_file_enabled=true \
    -e NEO4J_apoc_import_file_use__neo4j__config=true \
    -e NEO4JLABS_PLUGINS='["apoc", "graph-data-science"]' \
	--name neo4j \
	neo4j

