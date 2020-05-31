# -*- coding: utf-8 -*-

import sys
 
from pyspark import SparkContext, SparkConf, SQLContext

from pyspark.sql.types import StructType, StructField, FloatType, IntegerType

from pyspark.ml.feature import VectorAssembler, StringIndexer, MinMaxScaler
from pyspark.ml import Pipeline

from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import GBTClassifier, LogisticRegression, RandomForestClassifier





def createContext(appName= "Practica 4: Irene Bejar Maldonado"):
    # create Spark context with Spark configuration  
    conf = SparkConf().setAppName(appName)  
    sc = SparkContext(conf=conf)
    # Set Log info to display only error messages
    sc.setLogLevel("ERROR")
    return sc





def createDataset(sc, path_data, path_header):
    # Read header file
    headers = sc.textFile(path_header).collect()
    for line in headers:
        if "@inputs" in line:
            header_list = [x.replace(",","").strip() for x in line.split()]

    # Delete '@inputs' from the list
    del header_list[0]
    # Append column 'class' at the end
    header_list.append("class")

    # Read data file
    sqlContext = SQLContext(sc)
    df = sqlContext.read.csv(path_data, header=False, sep=",", inferSchema=True)
    print(df.schema)
    # Rename columns
    for i, colname in enumerate(df.columns):
        df = df.withColumnRenamed(colname, header_list[i])

    df = df.select("PSSM_r2_3_V", "AA_freq_global_L", "PSSM_central_-1_W", "PSSM_r2_1_H", "PSSM_central_0_Y", "PSSM_r1_3_N", "class")
    print(df.schema)

    # Save the new small dataframe in a file
    df.write.csv('./filteredC.small.training', header=True)





def loadSmallDataset(sc, path='./filteredC.small.training'):
    sqlContext = SQLContext(sc)

    # Prepare data schema
    data_schema = StructType([
        StructField("PSSM_r2_3_V", FloatType(), True),
        StructField("AA_freq_global_L", FloatType(), True),
        StructField("PSSM_central_-1_W", FloatType(), True),
        StructField("PSSM_r2_1_H", FloatType(), True),
        StructField("PSSM_central_0_Y", FloatType(), True),
        StructField("PSSM_r1_3_N", FloatType(), True),
        StructField("class", IntegerType(), True)
    ])

    # Read the data using the schema
    df = sqlContext.read.csv(path, header=True, sep=",", schema=data_schema)

    print ("--------------------------------------------------------------\n" + 
            "Dataset loaded\n" +
            "--------------------------------------------------------------\n")

    return df





def preprocessingData(sc, dataset):
    # Transform all the feature columns into a single vector column
    assemblerInputs = ["PSSM_r2_3_V", "AA_freq_global_L", "PSSM_central_-1_W", "PSSM_r2_1_H", "PSSM_central_0_Y", "PSSM_r1_3_N"]
    vec_assembler = VectorAssembler(inputCols =assemblerInputs, outputCol='features')

    # Add class column
    # StringIndexer converts a single column into a index column (similar to a factor column in R)
    indexer = StringIndexer(inputCol="class", outputCol="label")

    # Normalization 
    scaler = MinMaxScaler(inputCol="features", outputCol="scaled_features")

    # Apply transformations
    pipeline = Pipeline (stages=[vec_assembler, indexer, scaler])
    final_data = pipeline.fit(dataset).transform(dataset).select("scaled_features", "label")

    print ("--------------------------------------------------------------\n" + 
            "Preprocessed data\n")
    final_data.printSchema()
    print ("--------------------------------------------------------------\n")

    return final_data





def RUS_data(dataset):
    # The data is unbalanced. The ratio for each class is:
    #  BEFORE Class 0=1375458 Class 1=687729 ratio=2
    #  AFTER  Class 0=688517 Class 1=687729 ratio=1

    class_0 = dataset.filter(dataset["label"] == 0)
    class_1 = dataset.filter(dataset["label"] == 1)
    ratio = float(class_0.count()/class_1.count())
    
    # Data has to be balanced. RUS is used for this.
    sampled_majority = class_0.sample(False, float(1/ratio))
    final_data = sampled_majority.unionAll(class_1)

    print ("--------------------------------------------------------------\n" + 
            "Dataset balanced using RUS\n" +
            "--------------------------------------------------------------\n")

    return final_data





def split_dataset(dataset, train_size, test_size):
    class_0 = dataset.filter(dataset["label"] == 0)
    class_1 = dataset.filter(dataset["label"] == 1)

    train_0, test_0 = class_0.randomSplit(weights=[train_size, test_size], seed=16)
    train_1, test_1 = class_1.randomSplit(weights=[train_size, test_size], seed=16)

    train = train_0.unionAll(train_1)
    test = test_0.unionAll(test_1)

    return train, test





def tune_model(estimator, param_grid, evaluator, train_data):
    tvs = TrainValidationSplit(estimator=estimator,
                           estimatorParamMaps=param_grid,
                           evaluator=evaluator,
                           # 80% of the data will be used for training, 20% for validation.
                           trainRatio=0.8,
                           seed=16)
    
    model = tvs.fit(train_data)
    
    # print results for each combination
    for i, item in enumerate(model.getEstimatorParamMaps()):
        grid = ["%s: %s" % (p.name, str(v)) for p, v in item.items()]
        print(grid, model.getEvaluator().getMetricName(),
              model.validationMetrics[i])


    return model.bestModel





def train_and_validation(estimator, train_data, test_data):
    model = estimator.fit(train_data)

    evaluator = BinaryClassificationEvaluator()
    auc = evaluator.evaluate(model.transform(test_data))

    print ("--------------------------------------------------------------\n" + 
            "Validation Result\n")
    print("AUC sobre el conjunto de test: ", auc)
    print ("--------------------------------------------------------------\n")





def train_logisticRegresion(train_data, file_name):
    lr = LogisticRegression(labelCol="label", featuresCol="scaled_features", seed=16)

    paramGrid = ParamGridBuilder()\
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .addGrid(lr.elasticNetParam, [0.5, 1.0])\
    .addGrid(lr.maxIter, [10, 15])\
    .build()

    evaluator = BinaryClassificationEvaluator(labelCol="label")

    best_model = tune_model(lr, paramGrid, evaluator, train_data)
    best_model.overwrite().save(file_name)

    return best_model





def train_randomForest(train_data, file_name):
    rf = RandomForestClassifier(labelCol="label", featuresCol="scaled_features", seed=16)

    paramGrid = ParamGridBuilder()\
    .addGrid(rf.numTrees, [10,15,20]) \
    .addGrid(rf.maxDepth, [3, 5])\
    .build()

    evaluator = BinaryClassificationEvaluator(labelCol="label")

    best_model = tune_model(rf, paramGrid, evaluator, train_data)
    best_model.overwrite().save(file_name)

    return best_model



def train_GBT(train_data, file_name):
    gbt = GBTClassifier(labelCol="label", featuresCol="scaled_features", seed=16)

    paramGrid = ParamGridBuilder()\
    .addGrid(gbt.maxIter, [5, 10, 15]) \
    .addGrid(gbt.maxDepth, [2, 3, 10])\
    .build()

    evaluator = BinaryClassificationEvaluator(labelCol="label")

    best_model = tune_model(gbt, paramGrid, evaluator, train_data)
    best_model.overwrite().save(file_name)

    return best_model

if __name__ == "__main__":
    
    spark_context = createContext() # Start Spark Session

    data = "/user/datasets/ecbdl14/ECBDL14_IR2.data"
    header = "/user/datasets/ecbdl14/ECBDL14_IR2.header"

    # createDataset(sc=spark_context, path_data=data, path_header=header)
    df = loadSmallDataset(sc=spark_context)
    df = preprocessingData(spark_context, df)
    df = RUS_data(df)
    train, test = split_dataset(df, 70.0, 30.0)


    # Descomentar para volver a entrenar
    # # Regresión logística
    # lr = train_logisticRegresion(train, "lr.model")

    # # Random Forest
    # rf = train_randomForest(train, "rf.model")

    # # GBT
    # gbt = train_GBT(train, "gbt.model")


    # Entrenamiento y validación del modelo
    gbt = GBTClassifier(labelCol="label", featuresCol="scaled_features", maxIter=15, maxDepth=10, seed=16)
    train_and_validation(gbt, train, test)

    spark_context.stop() #To finalize Spark session

