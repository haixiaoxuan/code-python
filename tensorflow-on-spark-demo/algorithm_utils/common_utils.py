# -*-coding=utf8-*-

import getpass


def get_abs_path(sc, path):
    """ 拿到 HDFS 绝对路径 """

    defaultFS = sc._jsc.hadoopConfiguration().get("fs.defaultFS")

    if path.startswith("hdfs://"):
        return path
    elif path.startswith("/"):
        return defaultFS + path
    else:
        return "{0}/user/{1}/{2}".format(defaultFS, getpass.getuser(), path)
