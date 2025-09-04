"""
Kafka to Iceberg streaming job (Spark structured streaming).

This module contains a scaffold for a Spark Structured Streaming
application that reads events from Kafka topics, performs basic
transformations and writes them into Apache Iceberg tables.  Iceberg
provides ACID transactions, schema evolution and time travel on top
of cloud object stores such as Amazon S3.  See
https://iceberg.apache.org/ for more information.

To run this job, submit it with Spark using the appropriate Iceberg
extensions and catalog configuration.  For example:

.. code-block:: bash

    spark-submit \
        --packages org.apache.iceberg:iceberg-spark-runtime-3.3_2.12:1.4.0 \
        --conf spark.sql.catalog.my_catalog=org.apache.iceberg.spark.SparkCatalog \
        --conf spark.sql.catalog.my_catalog.type=hadoop \
        --conf spark.sql.catalog.my_catalog.warehouse=s3://constellation/warehouse \
        neuros/etl/iceberg_streaming_job.py \
        --bootstrap-servers kafka:9092 \
        --topics raw.eeg.dev0,raw.video.cam0 \
        --catalog my_catalog \
        --database raw \
        --table events_raw

This script is intentionally lightweight and focuses on the essential
Spark streaming constructs; production deployments should add proper
checkpointing, watermarking and error handling.
"""
from __future__ import annotations

import argparse
import json
import logging
from typing import Iterable

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, LongType, IntegerType, MapType

logger = logging.getLogger(__name__)


def build_schema() -> StructType:
    """Define a Spark schema matching the KafkaEvent JSON structure."""
    return StructType(
        [
            StructField("subject_id", StringType()),
            StructField("session_id", StringType()),
            StructField("device_id", StringType()),
            StructField("modality", StringType()),
            StructField("timestamp_ns", LongType()),
            StructField("seq", IntegerType()),
            StructField("payload", StringType()),
            StructField("meta", MapType(StringType(), StringType())),
        ]
    )


def run_streaming_job(
    bootstrap_servers: str,
    topics: Iterable[str],
    catalog: str,
    database: str,
    table: str,
) -> None:
    """Launch the Spark Structured Streaming job.

    Parameters
    ----------
    bootstrap_servers:
        Comma separated list of Kafka brokers.
    topics:
        Iterable of Kafka topic names to subscribe to.
    catalog:
        Name of the Iceberg catalog configured in Spark.
    database:
        Name of the database/schema to write to.
    table:
        Name of the Iceberg table to append events to.
    """
    spark = (
        SparkSession.builder.appName("neurOSIcebergStreaming")
        .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
        # Catalog configuration must be provided externally via submit args or default Hadoop config
        .getOrCreate()
    )

    logger.info("Starting streaming job on topics: %s", topics)
    # Read from Kafka
    df_raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", bootstrap_servers)
        .option("subscribe", ",".join(topics))
        .option("startingOffsets", "earliest")
        .load()
    )

    # Convert the binary value column containing JSON into columns
    schema = build_schema()
    df_parsed = df_raw.select(
        from_json(col("value").cast("string"), schema).alias("event")
    ).select("event.*")

    # Write to Iceberg table in append mode
    output_table = f"{catalog}.{database}.{table}"
    query = (
        df_parsed.writeStream.format("iceberg")
        .outputMode("append")
        .option("path", output_table)
        .option("checkpointLocation", f"/tmp/iceberg/checkpoints/{table}")
        .start()
    )

    query.awaitTermination()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Kafka to Iceberg streaming job")
    parser.add_argument("--bootstrap-servers", required=True)
    parser.add_argument("--topics", required=True, help="Comma separated list of topics")
    parser.add_argument("--catalog", required=True, help="Iceberg catalog name")
    parser.add_argument("--database", required=True, help="Database/schema name")
    parser.add_argument("--table", required=True, help="Table name")
    args = parser.parse_args(argv)

    run_streaming_job(
        bootstrap_servers=args.bootstrap_servers,
        topics=args.topics.split(","),
        catalog=args.catalog,
        database=args.database,
        table=args.table,
    )


if __name__ == "__main__":
    main()