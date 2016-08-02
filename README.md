# spark-sklearn-airbnb-predict
Code example to predict prices of Airbnb vacation rentals, using scikit-learn on Spark.

The [Jupyter notebook in this repo](https://github.com/mapr-demos/spark-sklearn-airbnb-predict/blob/master/python_scikit_airbnb.ipynb) contains examples to run regression estimators on the [Inside Airbnb](http://insideairbnb.com/get-the-data.html) listings dataset from San Francisco.  The target variable is the price of the listing.  To speed up the hyperparameter search, the notebook shows examples that use the spark-sklearn package to distribute GridSearchCV across nodes in a Spark cluster.  This provides a much faster way to search and can lead to better results.

To run the scikit-learn examples (without Spark) the following packages are required:
* Python 2
* Pandas
* NumPy
* scikit-learn (0.17 or later)

These can be installed on the [MapR Sandbox](https://www.mapr.com/products/mapr-sandbox-hadoop).

To run the scikit-learn examples with Spark, the following packages are required on each machine:
* All of the above packages
* Spark (1.5 or later)
* [spark-sklearn](https://github.com/databricks/spark-sklearn) -- follow the installation instructions there

You can run this on a MapR cluster by following one of these methods:
* Use the [MapR Sandbox](https://www.mapr.com/products/mapr-sandbox-hadoop), which comes with Spark pre-installed.  You must install Pandas, NumPy and scikit-learn.
* If you have multiple machines available, use the [MapR Community Edition](https://www.mapr.com/products/hadoop-download) and install the mapr-spark package on each machine, following the [Spark on YARN documentation](http://maprdocs.mapr.com/51/#Spark/SparkonYARN.html)

Run the script with:

```MASTER=yarn-client /opt/mapr/spark/spark-1.5.2/bin/spark-submit --num-executors=4 --executor-cores=8 python_scikit_airbnb.py```

(setting num-executors and executor-cores to suit your environment)

The file ```classify.py``` in this repo contains an example of classification on the same dataset, using ```reviews.csv``` and text analysis.

and of course... have fun!

