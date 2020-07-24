#!/bin/sh

spark-shell \
	--jars /home/areias/Documents/DataScience/graphs/neo4j-spark-connector-2.4.5-M1.jar \
#	--conf spark.neo4j.url=bolt://172.17.0.2:7687 \
#	--conf spark.neo4j.user=neo4j \
#	--conf spark.neo4j.password=test