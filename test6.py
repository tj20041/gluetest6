import sys
import logging
from pyspark.context import SparkContext
from pyspark.sql.functions import col, from_json, explode
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, DoubleType
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("glue_json_ingestion")

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

logger.info("Simulating incoming streaming Kafka payloads containing raw JSON events...")

raw_payloads = [
    ('{"batch_id": "B001", "records": [{"transaction_id": "TXN-9821A", "amount": 145.50, "currency": "USD"}]}',),
    ('{"batch_id": "B002", "records": [{"transaction_id": "TXN-4421B", "amount": 29.99, "currency": "EUR"}, {"transaction_id": "TXN-8812C", "amount": 499.00, "currency": "USD"}]}',)
]

raw_df = spark.createDataFrame(raw_payloads, ["payload_string"])

# FIX: transaction_id is alphanumeric (e.g. "TXN-9821A") in the actual upstream JSON payloads,
# so it must be declared as StringType. It was previously declared as IntegerType, which caused
# Spark's from_json to silently null out every transaction_id (since the cast to Integer failed),
# which in turn tripped the downstream null-check and raised a ValueError on every run.
nested_item_schema = StructType([
    StructField("transaction_id", StringType(), True),
    StructField("amount", DoubleType(), True),
    StructField("currency", StringType(), True)
])

root_payload_schema = StructType([
    StructField("batch_id", StringType(), True),
    StructField("records", ArrayType(nested_item_schema), True)
])

logger.info("Parsing raw JSON strings into structured schemas...")

try:
    parsed_df = raw_df.withColumn("parsed_data", from_json(col("payload_string"), root_payload_schema))
    exploded_df = parsed_df.select(
        col("payload_string"),
        col("parsed_data.batch_id").alias("batch_id"),
        explode(col("parsed_data.records")).alias("record")
    )

    logger.info("Validating strict non-null transaction IDs...")
    # Now that the schema correctly declares transaction_id as StringType, alphanumeric IDs
    # such as "TXN-9821A" parse successfully instead of being silently nulled out by from_json.
    clean_metrics = exploded_df.select(
        col("payload_string"),
        col("batch_id"),
        col("record.transaction_id").alias("tx_id"),
        col("record.amount").alias("amount")
    )

    # Trigger evaluation and surface per-record diagnostics on failure instead of only a bulk count,
    # so future schema-drift issues are diagnosable (per recommendation in the root-cause analysis).
    invalid_records_df = clean_metrics.filter(col("tx_id").isNull())
    invalid_count = invalid_records_df.count()
    if invalid_count > 0:
        sample_failed_payloads = [row["payload_string"] for row in invalid_records_df.select("payload_string").distinct().limit(5).collect()]
        logger.error(
            "Data contract violation: Found %s null transaction IDs after cast. Sample failing raw payloads: %s",
            invalid_count,
            sample_failed_payloads
        )
        raise ValueError(f"Corrupted records encountered: {invalid_count} records failed schema validation")

    clean_metrics.select("batch_id", "tx_id", "amount").show()
    job.commit()
except Exception as e:
    logger.error("Glue job failed during JSON parsing/validation stage: %s", str(e))
    try:
        logger.error("raw_df schema: %s", raw_df.schema.simpleString())
    except Exception:
        pass
    raise
