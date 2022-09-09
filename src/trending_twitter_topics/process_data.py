import os

import pyspark
import pyspark.sql.functions as sf


def load_json_data(spark, dir: str) -> pyspark.sql.dataframe.DataFrame:
    data: pyspark.sql.dataframe.DataFrame = None
    file_list = []
    for file in os.listdir(dir):
        file_list.append(os.path.join(dir, file))
    return spark.read.json(file_list)


def filter_cols(
    spark_df: pyspark.sql.dataframe.DataFrame, columns: list
) -> pyspark.sql.dataframe.DataFrame:
    return spark_df.select(columns)


def rename_cols(
    spark_df: pyspark.sql.dataframe.DataFrame, mapping: dict
) -> pyspark.sql.dataframe.DataFrame:
    for old_name, new_name in mapping.items():
        spark_df = spark_df.withColumnRenamed(old_name, new_name)
    return spark_df


def convert_to_datecol(
    spark_df: pyspark.sql.dataframe.DataFrame, spark, col: str, format: str
) -> pyspark.sql.dataframe.DataFrame:
    spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")
    df_datecol_converted = spark_df.withColumn(
        col, sf.to_timestamp(col, format=format)
    )
    return df_datecol_converted


def truncate_date(
    spark_df: pyspark.sql.dataframe.DataFrame, time_unit: str, col: str
) -> pyspark.sql.dataframe.DataFrame:
    return spark_df.withColumn(
        time_unit, sf.date_trunc(time_unit, sf.col(col))
    )


def explode_group_count(
    spark_df: pyspark.sql.dataframe.DataFrame,
    explode_col: str,
    groupby_cols: list,
) -> pyspark.sql.dataframe.DataFrame:

    return (
        spark_df.withColumn(explode_col, sf.explode(explode_col))
        .groupBy(groupby_cols)
        .count()
    )
