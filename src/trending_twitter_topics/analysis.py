import pyspark
import pyspark.sql.functions as sf
from pyspark.sql import Window


def add_change_per_time_unit(
    spark_df: pyspark.sql.dataframe.DataFrame,
    col_for_stats: str,
    partition_col: str,
    time_col: str,
    window_size: int = 1,
) -> pyspark.sql.dataframe.DataFrame:

    roll_window = Window.partitionBy(partition_col).orderBy(time_col)

    return (
        spark_df
        # add value of nth previous row
        .withColumn(
            "n_previous_value",
            sf.lag(col_for_stats, window_size).over(roll_window),
        )
        # add relative change
        .withColumn(
            "rel_change",
            (sf.col(col_for_stats) - sf.col("n_previous_value"))
            / sf.col("n_previous_value"),
        )
    )


def set_abs_thresholds_for_rel_change(
    spark_df: pyspark.sql.dataframe.DataFrame,
    col: str,
    prev_value_col: str,
    min_abs_value: int = 1,
    min_prev_abs_value: int = 1,
    rel_change_col: str = "rel_change",
) -> pyspark.sql.dataframe.DataFrame:
    return (
        spark_df
        # change rel_change to NaN if abs value of that row is < min_abs_value
        .withColumn(
            rel_change_col,
            sf.when(sf.col(col) >= min_abs_value, sf.col(rel_change_col)),
        )
        # change rel_change to NaN if abs value of previous nth row is < min_abs_nth_previous
        .withColumn(
            rel_change_col,
            sf.when(
                sf.col(prev_value_col) >= min_prev_abs_value,
                sf.col(rel_change_col),
            ),
        )
    )
