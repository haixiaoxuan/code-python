#!-*-coding=utf8-*-

def write_data_to_cluster(sc, save_path, data, is_text_file=True):
    """ 写数据到 hdfs
        @:param is_text_file 是否是文本文件，默认是文本文件采用 utf8 编码
    """
    Path = sc._gateway.jvm.org.apache.hadoop.fs.Path
    FileSystem = sc._gateway.jvm.org.apache.hadoop.fs.FileSystem
    Configuration = sc._gateway.jvm.org.apache.hadoop.conf.Configuration
    fs = FileSystem.get(Configuration())
    output = fs.create(Path(save_path))
    if is_text_file:
        output.write(bytearray(data, encoding="utf8"))
    else:
        output.write(data)
    output.close()


def save_df(df, input_dict):
    d_type = input_dict.get("type")
    if (d_type == "hive"):
        database = input_dict.get("database")
        table = input_dict.get("table")
        table_name = database + "." + table
        save_data_to_hive(df, table_name)


def save_data_to_hive(df, table_name):
    """ 将 df 保存为 hive 表 """
    df.write.saveAsTable(table_name)
