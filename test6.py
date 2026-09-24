import sys
import logging
from pyspark.context import SparkContext
from pyspark.sql.functions import col, from_json, explode
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType, DoubleType
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

# Fixed schema definition: transaction_id is a string identifier (e.g. 'TXN-9821A'),
# not a numeric value, so it must be declared as StringType to match the actual
# Kafka JSON payload contract (see AWS Glue ETL Pipelines Cookbook, Pipeline 2).
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

parsed_df = raw_df.withColumn("parsed_data", from_json(col("payload_string"), root_payload_schema))
exploded_df = parsed_df.select(
    col("payload_string"),
    col("parsed_data.batch_id").alias("batch_id"),
    explode(col("parsed_data.records")).alias("record")
)

logger.info("Validating strict non-null transaction IDs...")
# With the corrected StringType schema, transaction_id no longer nulls out on cast,
# but this validation is retained as a defensive data-quality gate for genuine
# upstream data issues (e.g. malformed or missing transaction_id fields).
clean_metrics = exploded_df.select(
    col("payload_string"),
    col("batch_id"),
    col("record.transaction_id").alias("tx_id"),
    col("record.amount").alias("amount"),
    col("record.currency").alias("currency")
)

# Trigger evaluation
invalid_records_df = clean_metrics.filter(col("tx_id").isNull())
invalid_count = invalid_records_df.count()
if invalid_count > 0:
    # Log a sample of the offending raw payloads/batch_ids so schema or data
    # contract mismatches are diagnosable directly from CloudWatch logs without
    # needing to re-run the job with additional debug instrumentation.
    sample_offending_rows = invalid_records_df.select("batch_id", "payload_string").limit(5).collect()
    for row in sample_offending_rows:
        logger.error(
            "Offending record - batch_id=%s payload_string=%s",
            row["batch_id"],
            row["payload_string"]
        )
    logger.error("Data contract violation: Found %s null transaction IDs after cast", invalid_count)
    raise ValueError(f"Corrupted records encountered: {invalid_count} records failed schema validation")

clean_metrics.drop("payload_string").show()
job.commit()
