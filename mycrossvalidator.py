#!/usr/bin/env python
# coding: utf-8

# ## Extending method  
# * https://stackoverflow.com/questions/2705964/how-do-i-extend-a-python-module-adding-new-functionality-to-the-python-twitter  
# * https://stackoverflow.com/questions/33511529/create-custom-cross-validation-in-spark-ml  
# * https://stackoverflow.com/questions/32331848/create-a-custom-transformer-in-pyspark-ml  

# In[372]:


import findspark
findspark.init()

from pyspark import keyword_only
from multiprocessing.pool import ThreadPool
from pyspark.ml.tuning import CrossValidator, _parallelFitTasks, CrossValidatorModel
import numpy as np
from pyspark.sql.functions import rand

class MyCrossValidator(CrossValidator):

    def _fit(self, dataset):
        est = self.getOrDefault(self.estimator)
        epm = self.getOrDefault(self.estimatorParamMaps)
        numModels = len(epm)
        eva = self.getOrDefault(self.evaluator)
        
        folds = dataset.select('fold').distinct().collect()
        nFolds = len(folds)
        
        metrics = [0.0] * numModels

        pool = ThreadPool(processes=min(self.getParallelism(), numModels))
        subModels = None
        
        allMetrics=[[None for i in range(nFolds)] for j in range(numModels)]
        
        collectSubModelsParam = self.getCollectSubModels()
        if collectSubModelsParam:
            subModels = [[None for j in range(numModels)] for i in range(nFolds)]

        for i in range(nFolds):
           
            validation = dataset.filter((dataset['fold']==i) & (dataset['test']==1)).cache()
            train = dataset.filter((dataset['fold']==i) & (dataset['test']==0)).cache()

            tasks = _parallelFitTasks(est, train, eva, validation, epm, collectSubModelsParam)
            for j, metric, subModel in pool.imap(lambda f: f(), tasks):
            
                allMetrics[j][i] = metric
                
                metrics[j] += (metric / nFolds)
                if collectSubModelsParam:
                    subModels[i][j] = subModel

            validation.unpersist()
            train.unpersist()

        if eva.isLargerBetter():
            bestIndex = np.argmax(metrics)
        else:
            bestIndex = np.argmin(metrics)
        bestModel = est.fit(dataset, epm[bestIndex])
        

        return (self._copyValues(CrossValidatorModel(bestModel, metrics, subModels)), allMetrics)

