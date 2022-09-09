import pyspark
from pyspark.sql import DataFrame

from trending_twitter_topics.analysis import (
    add_change_per_time_unit,
    set_abs_thresholds_for_rel_change,
)
from trending_twitter_topics.process_data import (
    convert_to_datecol,
    explode_group_count,
    filter_cols,
    load_json_data,
    rename_cols,
    truncate_date,
)


def pipe(self, func, *args, **kwargs):
    return func(self, *args, **kwargs)


DataFrame.pipe = pipe


def run_pipeline():
    data_dir = "data/Tweets/"
    master = "local[2]"

    spark = pyspark.sql.SparkSession.builder.master(master).getOrCreate()

    return (
        (
            load_json_data(spark, data_dir)
            .pipe(filter_cols, ["created_at", "entities.hashtags.text"])
            .pipe(rename_cols, {"text": "hashtags"})
            .pipe(
                convert_to_datecol,
                spark,
                "created_at",
                "EEE MMM dd HH:mm:ss ZZZZZ yyyy",
            )
            .pipe(truncate_date, "hour", "created_at")
            .pipe(explode_group_count, "hashtags", ["hour", "hashtags"])
            .pipe(add_change_per_time_unit, "count", "hashtags", "hour", 1)
            .cache()
            .pipe(
                set_abs_thresholds_for_rel_change,
                "count",
                "n_previous_value",
                50,
                5,
            )
        )
        .toPandas()
        .assign(
            max_per_hour=lambda df: df.groupby("hour")["rel_change"].transform(
                "max"
            )
        )
        .loc[lambda df: df["rel_change"] == df["max_per_hour"]]
        .sort_values(["hour"])
    )


if __name__ == "__main__":
    run_pipeline()
