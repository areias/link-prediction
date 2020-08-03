#!/usr/bin/env python
# coding: utf-8

# In[48]:
import numpy as np

# this could be rewritten to take in the function as an argument
def binary_operator(name,u,v):
    
    def operator_hadamard(u, v):
        return u * v

    def operator_l1(u, v):
        return np.abs(u - v)

    def operator_l2(u, v):
        return (u - v) ** 2

    def operator_avg(u, v):
        return (u + v) / 2.0

    if name=="hadamard":
        return operator_hadamard(u,v)
    elif name=="l1":
        return operator_l1(u,v)
    elif name=="average":
        return operator_avg(u,v)
    elif name=="l2":
        return operator_l2(u,v)
    


# In[49]:


import pandas as pd


# In[54]:


from py2neo import  Graph, Node

graphdb = Graph(scheme="bolt", host="localhost", port=7687, secure=False, auth=('neo4j', 'test'))


# In[ ]:


# create a graph projection
graphdb.run("""CALL gds.graph.create('early_graph',
    'Author', 
    {
        CO_AUTHOR_EARLY: {
                type: 'CO_AUTHOR_EARLY',
                orientation: 'UNDIRECTED'
                }
                }
                )""")


# In[ ]:


# create a graph projection
graphdb.run("""CALL gds.graph.create('late_graph',
    'Author', 
    {
        CO_AUTHOR: {
                type: 'CO_AUTHOR',
                orientation: 'UNDIRECTED'
                }
                }
                )""")


# In[ ]:


import pyspark
sc = pyspark.SparkContext(appName="link_inparts")


# In[7]:


from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("link_inparts").getOrCreate()


import gensim.models
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import StructField, StructType,IntegerType

def apply_node2vec_features(data, graph_name, walk_length, num_walks, dimensions, 
                            window_size, p, q, num_iter, workers, operator_name,
                            output_col_name):
    
    params = {
    "pairs": [{"node1": row["node1"], "node2": row["node2"]}
    for row in data.collect()],
    "steps": walk_length,
    "walks": num_walks,
    "size": dimensions,
    "graph_name": graph_name,
    "mode":"node2vec",
    "inOut":q,
    "return":p
    }

    query=("""
    UNWIND $pairs as pair
    MATCH (p:Author) WHERE id(p) = pair.node1 OR id(p) = pair.node2
    WITH DISTINCT p
    CALL gds.alpha.randomWalk.stream($graph_name,{
        start: id(p),
        steps: $steps,
        walks: $walks,
        mode: $mode,
        inOut: $inOut,
        return: $return  
    })
    YIELD nodeIds
    RETURN [id in nodeIds | toString(id)] as walks
    """)

    random_walks=graphdb.run(query, params).to_series()
    
    model=gensim.models.Word2Vec(random_walks, sg=1, window=window_size, size=dimensions, min_count=1,
                                 workers=workers,iter=num_iter)

    vectors=[{"node1":row["node1"],
            "node2": row["node2"],
            output_col_name: Vectors.dense(
                binary_operator(operator_name, model.wv[str(row["node1"])], model.wv[str(row["node2"])]))
            } for row in data.collect()]
    
    schema = StructType([
        StructField('node1', IntegerType()),
        StructField('node2', IntegerType()),
        StructField(output_col_name, VectorUDT())])

    features=spark.createDataFrame(vectors, schema)
    return data.join(features, ["node1", "node2"])


# In[ ]:


dimensions = 128
num_walks = 10
walk_length = 80
window_size = 10
num_iter = 1
workers = 2
operator_name="l1"


import pickle
from sklearn.model_selection import ParameterGrid
from pyspark.ml.feature import RFormula
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import mycrossvalidator as mycv
from pyspark.sql.functions import lit



for q in [0.25,0.5,1,2,4]:


    grid = [{'p': [0.25,0.5,1,2,4],'q': [q]}]


    test_df=pd.read_csv("/home/areias/Documents/DataScience/graphs/data/testdf.csv")
    training_df=pd.read_csv("/home/areias/Documents/DataScience/graphs/data/trainingdf.csv")

    training_data=spark.createDataFrame(training_df)
    test_data=spark.createDataFrame(test_df)


    for item in enumerate(ParameterGrid(grid)):
        model_name="model"+str(item[0])
        training_data = apply_node2vec_features(training_data, 'early_graph', walk_length, num_walks, dimensions, 
                                window_size, item[1]['p'], item[1]['q'], num_iter, workers, operator_name,
                                model_name)
        test_data = apply_node2vec_features(test_data, 'late_graph', walk_length, num_walks, dimensions, 
                                window_size, item[1]['p'], item[1]['q'], num_iter, workers, operator_name,
                                model_name)


    rForm = RFormula()

    params = ParamGridBuilder()\
        .addGrid(rForm.formula, ["label ~ model"+str(i) for i in range(5)])\
        .build()


    rf = RandomForestClassifier(labelCol="label", 
            featuresCol="features",
            numTrees=30, maxDepth=10)

    stages=[rForm, rf]
    pipeline=Pipeline().setStages(stages)


    evaluator = BinaryClassificationEvaluator()\
                .setMetricName("areaUnderROC")\
                .setRawPredictionCol("prediction") \
                .setLabelCol("label")


    cv = mycv.MyCrossValidator() \
        .setEstimator(pipeline) \
        .setEvaluator(evaluator) \
        .setEstimatorParamMaps(params) \
        .setCollectSubModels(True)


    training_data=training_data.withColumn("fold",lit(0))
    training_data=training_data.withColumn("test",lit(0))


    test_data=test_data.withColumn("fold",lit(0))
    test_data=test_data.withColumn("test",lit(1))


    all_df=training_data.union(test_data)

    mycvfitted, foldstats = cv.fit(all_df)


    filename="/home/areias/Documents/DataScience/graphs/model_q"+str(q).replace(".","")+".pkl"

    with open(filename, 'wb') as f:
        pickle.dump(mycvfitted.avgMetrics, f)