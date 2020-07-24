

# Graphs and Machine Learning

What we can do 

1. Node classification or the labelling of missing attributes
2. Link prediction
3. Community detection or identifying subgraphs
4. Graph labelling 



How do we do it

1. Graph topology measures
2. Graph embeddings
3. Graph neural networks



4. Applications

* Recommender systems
* Fraud
* Drug discovery



## Neo4j and Python
Neo4j, Docker, Conda and Cypher
Python library networkX


## Graph topology measures
1. Path Finding
2. Centrality
3. Community detection  


## Graph embeddings
Representing graphs in a way suitable for machine learning taks, Reducing dimensionality
* node2vec
* struct2vec
* GraphSAGE
* DeepWalk
* DeepGL
Types of graph embedding to use depneds on type of graph (one node type or many) or what aspect of graph trying to represent (vertex, edge or whole graph)





## Graph neural networks
Graph structured data
Distributed vector representation 
Different network architechtures
	* Convolutional neural networks
	* Gated graph neural networks
Deep Graph Library
StellarGraph https://www.stellargraph.io/library



## Graphs and data viz
Graph visualization as a way to understand neural network predictions
Graphs and explainable AI models https://www.kdnuggets.com/2019/04/machine-learning-graph-analytics.html



## Graph datasets
 [Stanford Large Network Dataset Collection] https://snap.stanford.edu/data/





## Start zeppelin

https://zeppelin.apache.org/docs/0.7.0/install/docker.html

https://zeppelin.apache.org/download.html


docker run \
	-u `id -u $USER` \
	-p 8080:8080 \
	--rm \
	-v /home/areias/Documents/DataScience/graphs/logs:/logs \
	-v /home/areias/Documents/DataScience/graphs/notebook:/notebook \
	-v home/areias/Documents/DataScience/graphs/data:/zeppelin/data \
	-e ZEPPELIN_LOG_DIR='/logs' \
	-e ZEPPELIN_NOTEBOOK_DIR='/notebook' \
	--name zeppelin \
	apache/zeppelin:0.9.0


## Start Neo4j

https://neo4j.com/docs/operations-manual/current/docker/introduction/

docker run \
	-u `id -u $USER`  \
	--publish=7473:7473 \
	--publish=7474:7474 \
	--publish=7687:7687 \
	--rm \
	--volume=/home/areias/Documents/DataScience/graphs/data:/data \
    --volume=/home/areias/Documents/DataScience/graphs/logs:/logs \
	--env NEO4J_AUTH=neo4j/test \
	--env NEO4JLABS_PLUGINS=["apoc","graphql"]
	--env NEO4J_dbms_memory_pagecache_size=4G \
	--name neo4j \
	neo4j




## connect zepelin to neo4j

https://zeppelin.apache.org/docs/0.8.0/interpreter/neo4j.html


## load an example database




docker exec -it zeppelin bash


everytime run docker containers it then sets permissions to data and log folders to other 
https://denibertovic.com/posts/handling-permissions-with-docker-volumes/


https://www.linux.org/threads/file-permissions-chmod.4124/
Think of the chmod command actually having the following syntax...

chmod owner group world FileName

4 read (r)
2 write (w)
1 execute (x)

https://stackoverflow.com/questions/37299077/neo4j-importing-local-csv-file#37299535
dbms.security.allow_csv_import_from_file_urls=true


counlnt connect zeppelin to neo4j



create network
spin up neo4j container on network
spin up jupyter notebook container on network
install py2neo
connect jupyter to neo4j



ok what finally worked!!! 
https://github.com/neo4j-contrib/neo4j-jdbc
https://blog.armbruster-it.de/2016/11/integrate-neo4j-apache-zeppelin/

## Apoc procedures to load json data
https://github.com/neo4j-contrib/neo4j-apoc-procedures


## Node clasification examples

predict subject for a paper

https://www.kdnuggets.com/2019/08/neighbours-machine-learning-graphs.html

Graph enhanced ML workflow

## Link prediction 

predict future colaboration between authors

v10 citation Dataset from 

https://www.aminer.org/citation


can find experts 



interpret
create
name neo4j2
choose jdbc interpreter




## Spark neo4j connector

https://www.nielsdejong.nl/neo4j%20projects/2020/05/11/neo4j-spark-connector-databricks.html

https://github.com/neo4j-contrib/neo4j-spark-connector/blob/master/README.md

https://neo4j.com/blog/neo4j-3-0-apache-spark-connector/

https://towardsdatascience.com/building-a-graph-data-pipeline-with-zeppelin-spark-and-neo4j-8b6b83f4fb70

https://neo4j.com/blog/neo4j-3-0-apache-spark-connector/

http://spark.apache.org/docs/latest/configuration.html

https://spark-packages.org/package/neo4j-contrib/neo4j-spark-connector


$SPARK_HOME/bin/spark-shell \
--conf spark.neo4j.bolt.password= \
--packages neo4j-contrib:neo4j-spark-connector:1.0.0-RC1,\
graphframes:graphframes:0.1.0-spark1.6

$SPARK_HOME/bin/spark-shell --jars neo4j-spark-connector_2.11-full-2.4.5-M1.jar

$SPARK_HOME/bin/spark-shell --packages neo4j-contrib:neo4j-spark-connector:2.4.5-M1


spark.neo4j.bolt.encryption true
spark.databricks.delta.preview.enabled true
spark.neo4j.bolt.password change
spark.neo4j.bolt.user neo4j
spark.neo4j.bolt.url bolt+routing://f1337.databases.neo4j.io:7687


val conf = new SparkConf()
             .setMaster("local[2]")
             .setAppName("CountingSheep")
val sc = new SparkContext(conf)


Any jar can be added to spark-shell by using the --jars command:


downloaded jar file https://github.com/neo4j-contrib/neo4j-spark-connector/releases/tag/2.4.5-M1

see start-spark-neo4j-connector.sh

import org.neo4j.spark._

val neo = Neo4j(sc)

val df = neo.cypher("MATCH (n:Author) RETURN n.name LIMIT 25").loadDataFrame

df.printSchema()

IT FUCKING WORKS!



now you gotta try starting it first
then set conf inside

works!

but gotta put the jar file somewhere zepellin finds it



import org.apache.spark.{SparkContext, SparkConf}

sc.stop()
val conf = new SparkConf().set("spark.neo4j.url", "bolt://172.17.0.2:7687").set("spark.neo4j.user","neo4j").set("spark.neo4j.password","test")

val sc = new SparkContext(conf)

import org.neo4j.spark._

val neo = Neo4j(sc)

val df = neo.cypher("MATCH (n:Author) RETURN n.name LIMIT 25").loadDataFrame

df.printSchema()

it works!!!

added the jar as a dependency in spark interpreter












