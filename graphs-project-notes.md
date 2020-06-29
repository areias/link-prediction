

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


docker run -p 8080:8080 --rm -v /home/areias/Documents/Babjob/logs:/logs -v /home/areias/Documents/Babajob/notebook:/notebook -v /home/areias/Documents/Babajob:/zeppelin/Babajob -e ZEPPELIN_LOG_DIR='/logs' -e ZEPPELIN_NOTEBOOK_DIR='/notebook' --name zeppelin apache/zeppelin:0.9.0


## start neo4j

https://neo4j.com/docs/operations-manual/current/docker/introduction/

docker run --publish=7473:7473 --publish=7474:7474 --publish=7687:7687 -v=/home/areias/Documents/DataScience/neo4j/:/home --env NEO4J_AUTH=neo4j/test neo4j





## connect zepelin to neo4j

https://zeppelin.apache.org/docs/0.8.0/interpreter/neo4j.html


## load an example database


