#!-*-coding=utf8-*-


def get_data(spark, input_dict, num_partitions):
    """
        get_train_data 支持三种数据源： hdfs的csv，hive ，postgre 数据库
        ------------------------------------------------------------------
        source      |       input_dict
        ------------------------------------------------------------------
        hdsf_csv    |     {"type":"hdfs","path":""}
        hive        |     {"type":"hive","database":"","table":""}
        postgresql  |     {"type":"postgre","url":"","table":"","username":"","password":""}
        ------------------------------------------------------------------
        :param num_partitions 指定分区数
    """

    input_type = input_dict.get("type")
    if input_type == "hdfs":  # 读取 HDFS
        path = input_dict.get("path")
        return read_hdfs_csv_data(spark, path)
    elif input_type == "hive":  # 读取 HIVE
        database = input_dict.get("database")
        table = input_dict.get("table")
        table_name = database + "." + table
        return read_hive_data(spark, table_name, num_partitions)
    elif input_type == "postgre":  # 读取 POSTGRESQL
        url = input_dict.get("url")
        table = input_dict.get("table")
        properties = {"user": input_dict.get("username"), "password": input_dict.get("password")}
        return read_db_data(spark, url, table, properties, num_partitions)


def read_hdfs_csv_data(spark, path, num_partitions=None,
                       header=True, inferSchema=True):
    """ 读取hdfs数据 """
    data_df = spark.read.csv(path, header=header, inferSchema=inferSchema)
    data_df = data_df if num_partitions is None else data_df.repartition(num_partitions)
    return data_df


def read_hive_data(spark, table_name, num_partitions=None):
    """ 读取 hive 表中的数据 """
    data_df = spark.sql("select * from {0}".format(table_name))
    data_df = data_df if num_partitions is None else data_df.repartition(num_partitions)
    return data_df


def read_db_data(spark, url, table, properties, num_partitions=None):
    """ 获取postgresql数据 """
    data_df = spark.read.jdbc(url=url, table=table, properties=properties)
    data_df = data_df if num_partitions is None else data_df.repartition(num_partitions)
    return data_df
