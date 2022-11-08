from pyspark.sql import SparkSession
import os
from configparser import ConfigParser
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql.functions import udf
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import ArrayType, IntegerType
from pyspark.ml.feature import StandardScaler
import sys
from pyspark.ml.feature import PCA
import pandas as pd
import numpy as np


def get_env(config_file_path: str):
    try:
        conf = ConfigParser()
        conf.read(config_file_path)
        return conf
    except Exception:
        print(Exception)
        sys.exit(-1)


def build_spark_session(master, sqlite_version):
    try:
        session = SparkSession.builder.master(master). \
            config('spark.jars.packages', f"org.xerial:sqlite-jdbc:{sqlite_version}").getOrCreate()
        session.conf.set("spark.sql.shuffle.partitions", 100)
        session.conf.set("spark.sql.execution.arrow.pyspark.enabled", True)
        return session
    except Exception:
        print(f'unable to create spark session. {Exception}')


@udf(ArrayType(IntegerType()))
def julian_to_date(x):
    try:
            
        dt = pd.to_datetime(float(x)-epoch, unit='D')
        return [int(dt.year), int(dt.month), int(dt.weekday())]
    
    except Exception:
        print(f"unable get julian date. {Exception}")


def set_env(tpl="env.tpl"):

    try:
        env = get_env(tpl)
        for i in env['vars']:
            os.environ[i] = env['vars'][i]
    except Exception:
        print(f"unable to set environment variables. {Exception}")


def get_df(spark, table):
    try:
        df = (spark.read.format('jdbc')
                   .options(url=f'jdbc:sqlite:/save/FPA.sqlite', dbtable=table, driver='org.sqlite.JDBC')
                   .load())
        return df
    except Exception:
        print(f"unable to get dataframe. {Exception} ")


def subset_transform(df, predictors, target, datetime_col):
    try:
        df_t = df.select(target, *predictors, *[julian_to_date(datetime_col)[i] for i in range(0, 3)]) \
            .toDF(target, *predictors, 'YEAR', 'MONTH', 'WEEKDAY')

        return df_t
    except Exception:
        print(f"unable to transform df. {Exception}")


def scaler(df):
    assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")
    df_features_target = assembler.transform(df)
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features",
                        withStd=True, withMean=False)
    scaling = scaler.fit(df_features_target)
    df_scaled = scaling.transform(df_features_target)
    return df_scaled


def how_many_pca(df, n_features):
    try:
        print("getting number of components")
        pca = PCA(k=n_features-1, inputCol="scaled_features")
        model = pca.fit(df)
        model.transform(df).collect()
        features_ev = model.explainedVariance
        ev = 0
        for i in range(len(features_ev)):
            ev = ev + features_ev[i]
            if ev > 0.8:
                k = i
                return k
        return -1
    except Exception:
        print(f"unable to get k due to {Exception}")




def get_pca_df(df_scaled, k):

    try:
        print("creating principal components df")
        pca = PCA(k=k, inputCol="scaled_features")
        pca.setOutputCol("pca_features")
        model = pca.fit(df_scaled)
        return model.transform(df_scaled)

    except Exception:
        print(f"unable to get components df. {Exception}")


if __name__ == "__main__":
    
    os.chdir(os.environ["VOLUME_PATH_HOST"])
    spark = build_spark_session(os.environ['SPARK_MASTER'], os.environ['SQLITE_VERSION'])
    df = get_df(spark, os.environ["TABLE"])
    epoch = pd.to_datetime(0, unit='s').to_julian_date()
    predictors = os.environ['PREDICTORS'].replace(" ", "").split(",")
    df_subset_transform = subset_transform(df, np.array(predictors), os.environ["TARGET"], os.environ["DATETIME_COL"])
    numeric_cols = predictors+['YEAR', 'MONTH', 'WEEKDAY']
    df_scaled = scaler(df_subset_transform)
    k_pca = how_many_pca(df_scaled, len(numeric_cols))
    df_pca = get_pca_df(df_scaled, k_pca)
    train, test = df_pca.randomSplit([0.8, 0.2], seed=1022)
    rf = RandomForestClassifier(featuresCol='pca_features', labelCol=os.environ["TARGET"], seed=1022)
    evaluator = MulticlassClassificationEvaluator(labelCol=os.environ["TARGET"], predictionCol="prediction",
                                                  metricName=os.environ["METRIC"])
    params = (ParamGridBuilder().addGrid(rf.numTrees, [100, 500, 1000]).build())
    cv = CrossValidator(estimator=rf, estimatorParamMaps=params, evaluator=evaluator, numFolds=4, seed=1022, parallelism=3)
    print('training model...')
    model = cv.fit(train)
    print('getting predictions')
    predictions = model.transform(test)
    accuracy = evaluator.evaluate(predictions)
    print(f'{os.environ["METRIC"]}: {accuracy}')
    model.save(os.environ["VOLUME_PATH_HOST"])
    md = model.bestModel.stages[-1]._java_obj.getMaxDepth()
    nt = model.bestModel.stages[-1]._java_obj.getNumTrees()
    print(f"best model ntrees: {nt}, maxdepth: {md}")
