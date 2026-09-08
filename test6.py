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

# Fixed schema definition: transaction_id is now correctly declared as StringType
# to match the actual alphanumeric values (e.g. 'TXN-9821A') in the upstream payloads.
# Previously it was IntegerType(), which caused Spark's from_json to silently coerce
# all transaction_id values to null, triggering the downstream null-check ValueError.
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
    col("parsed_data.batch_id").alias("batch_id"),
    explode(col("parsed_data.records")).alias("record")
)

logger.info("Validating strict non-null transaction IDs...")
clean_metrics = exploded_df.select(
    col("batch_id"),
    col("record.transaction_id").alias("tx_id"),
    col("record.amount").alias("amount")
)

# Trigger evaluation
invalid_count = clean_metrics.filter(col("tx_id").isNull()).count()
if invalid_count > 0:
    logger.error("Data contract violation: Found %s null transaction IDs after cast", invalid_count)
    # Surface the problematic rows in Glue CloudWatch logs before raising
    clean_metrics.filter(col("tx_id").isNull()).show(truncate=False)
    raise ValueError(f"Corrupted records encountered: {invalid_count} records failed schema validation")

clean_metrics.show()
job.commit()
