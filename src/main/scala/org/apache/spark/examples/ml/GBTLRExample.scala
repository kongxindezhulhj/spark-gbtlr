package org.apache.spark.examples.ml

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.gbtlr.{GBTLRClassificationModel, GBTLRClassifier}
import org.apache.spark.ml.classification.BinaryLogisticRegressionSummary
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{Row, SparkSession}

// scalastyle:off println


object GBTLRExample {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)

    val spark = SparkSession
      .builder()
      .master("local[2]")  //集群环境注释掉改行，且需要将后面的文件路径改成hdfs上面的路径
      .appName("gbtlr cv example")
      .getOrCreate()

    val startTime = System.currentTimeMillis()

    val dataset = spark.read.option("header", "true").option("inferSchema", "true")
      .option("delimiter", ";").csv("data/bank/bank-full.csv")

    //将字符串转换成数字下标 ref:https://cloud.tencent.com/developer/article/1172180
    val columnNames = Array("job", "marital", "education",
      "default", "housing", "loan", "contact", "month", "poutcome", "y")
    val indexers = columnNames.map(name => new StringIndexer()
      .setInputCol(name).setOutputCol(name + "_index"))
    val pipeline = new Pipeline().setStages(indexers)
    val data1 = pipeline.fit(dataset).transform(dataset)

    //重命名字段
    val data2 = data1.withColumnRenamed("y_index", "label")

    //将多列数据转化为单列的向量列 ref:https://blog.csdn.net/lichao_ustc/article/details/52688127
    val assembler = new VectorAssembler()
    assembler.setInputCols(Array("age", "job_index", "marital_index",
      "education_index", "default_index", "balance", "housing_index",
      "loan_index", "contact_index", "day", "month_index", "duration",
      "campaign", "pdays", "previous", "poutcome_index"))
    assembler.setOutputCol("features")

    val data3 = assembler.transform(data2)

    //将原RDD划分成两个RDD，分别作为训练集和测试集
    val data4 = data3.randomSplit(Array(4, 1))

    val gBTLRClassifier = new GBTLRClassifier()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setGBTMaxIter(10)
      .setLRMaxIter(100)
      .setRegParam(0.01)
      .setElasticNetParam(0.5)

    val model = gBTLRClassifier.fit(data4(0))

    /*//打印部分数据，与加载后的数据预测结果比对
    var df =model.transform(data4(1))
    df.select("age", "job", "probability", "prediction")
      .collect().take(10)
      .foreach {
        case Row(age: Int, job: String, probability: Vector, prediction: Double) =>
          println(s"($age, $job) --> prob=$probability, prediction=$prediction")
      }*/

    val summary = model.evaluate(data4(1))
    val endTime = System.currentTimeMillis()
    val auc = summary.binaryLogisticRegressionSummary
      .asInstanceOf[BinaryLogisticRegressionSummary].areaUnderROC

    println(s"Training and evaluating cost ${(endTime - startTime) / 1000} seconds")
    println(s"The model's auc: ${auc}")


    //模型保存
    model.write.overwrite.save("data/bank/gbtlr_model")

    //模型加载
    println("***************load the model*******************")
    var modelx = GBTLRClassificationModel.load("data/bank/gbtlr_model") //return a GBTLRClassificationModel instance
    val summaryx = modelx.evaluate(data4(1))

    val aucx = summaryx.binaryLogisticRegressionSummary
      .asInstanceOf[BinaryLogisticRegressionSummary].areaUnderROC

    println(s"The modelxxxxxxx's auc: ${aucx}")

    //获取预测概率值 probability[0]代表预测为0的概率 probability[1]代表预测为1的概率
    var dfx=modelx.transform(data4(1))
    //df.show(10)
    dfx.select("age", "job", "probability", "prediction")
      .collect().take(10)
      .foreach {
        case Row(age: Int, job: String, probability: Vector, prediction: Double) =>
          println(s"($age, $job) --> prob=$probability, prediction=$prediction")
      }

  }
}

// scalastyle:on println
