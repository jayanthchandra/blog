---
date: "2025-02-27T00:34:08+05:30"
title: "How we built a Scalable Data Platform"
tags: ["data", "platform"]
draft: false
showAuthorsBadges: false
---

Building and managing a data platform that is both scalable and cost-effective is a challenge many organizations face. We managed an extensive data lake with a lean data team and reduced our `Infra Cost by 70%`.

This article explores how we built it and the lessons we learned. Hopefully, some of these insights will be useful (or at least interesting!) for your own data platform projects, regardless of your team size.

## Our Data Landscape

We are a fintech startup helping SMEs raise capital from our platform where we provide diverse financial products ranging from Term Loan, Revenue Based Financing to Syndication, we face one unique data challenge: **Our data comes from everywhere**.

Our clients often manage their financial information in different ways, leading to data sources ranging from the structured data in our MongoDB and PostgreSQL databases to the semi-structured data found in marketplaces, Google Sheets, and various payment platforms.

Storing the data was only part of the equation. We needed to process and analyse it at scale, transform it into actionable intelligence that drive key business decisions. Our data and BI analysts play a crucial role in this process, requiring robust data tooling to effectively access, analyze, and visualize the data. From lead generation and risk analysis to payment alerting and reconciliation, data is at the heart of our operations, and our data platform needs to support these critical workflows throughout the entire application lifecycle.

## Our Initial Data Platform

![Initial Data Platform](/images/Initial_Data_Infra.png)
Our initial data platform was built on a foundation of tools that addressed our early-stage needs, but quickly became insufficient as we scaled. Data ingestion was primarily handled by Hevo, which leveraged Debezium Slots for capturing CDC events from our databases, oplog for MongoDB, and direct integration with Google Sheets. While Hevo simplified initial data capture, its data transformation capabilities were limited, primarily offering basic functionality like data key deletion and value formatting.

Revenue data from marketplaces was ingested through an RPA-driven process, with data being directly ingested into Google BigQuery (GBQ) as raw dumps. BigQuery (GBQ). While this approach was simpler, it came with high costs, as GBQ is priced based on the amount of data queried. Given that the data sizes for each table were in the order of 200-500GBs, the costs quickly escalated.

Furthermore, a significant portion of queries were executed directly against our live OLTP tables. This direct querying increased the load on our production databases, impacting performance and further contributing to cost increases.

In the early days, with a smaller team and less data, these engineering decisions were pragmatic and likely solved the immediate problems. However, as our company grew and data demands increased, it became clear that this solution was not scalable and could not meet our evolving requirements. This realization led to the creation of a new data team, with myself and my manager, Aankesh [^1], tasked with building a more robust and scalable data platform. We needed a platform that could handle the volume, variety, and complexity of our data, while also providing the necessary tools for efficient analysis and decision-making.

## Our New Data Platform

We implemented an ELT stack for our new data platform, leveraging cheap storage to prioritize raw data ingestion and subsequent in-warehouse transformations. We also strategically reused existing software components where they weren't being fully utilized, further optimizing our development efforts.

The platform's development was segmented into two layers: Data Ingestion and Storage & Compute.

### Data Ingestion Layer

![Data Ingestion](/images/Data_Ingestion.png)

- **Debezium:** Implemented for capturing CDC events from PostgreSQL and MongoDB, enabling real-time data replication.
- **Airflow:** Utilized to orchestrate manual data ingestion from sources like Google Sheets and CSV files.
- **Kafka & Kafka Connect:**
  - Formed the core of our streaming data pipeline.
  - Leveraged custom Single Message Transforms (SMTs) for specialized transformations.
  - Self-managed and hosted Kafka Connect cluster for fine-grained control.
  - Utilized **managed Confluent Cloud** for our Kafka Connect cluster, leveraging our existing infrastructure used for application pub-sub systems.
- **Sink Connectors:** Employed Kafka Connect Sink Connectors to deliver data to downstream destinations, including:
  - File storage (S3).
  - PostgreSQL for data replication.

### Storage & Compute Layer

![Data Processing](/images/Data_Platform.png)

- **Data Storage: Efficient and Cost-Effective Persistence**
  - All raw data, ingested from our diverse sources, is persisted in file storage (S3) in Parquet format. This choice offers significant advantages: Parquet's columnar storage optimizes query performance, and S3 provides cost-effective and highly durable storage.
- **Data Transformation and Quality: Robust Pipelines and Validation**
  - Airflow orchestrates dbt runs, enabling us to build modular, testable, and maintainable data transformation pipelines. dbt's transformation logic, expressed as SQL, simplifies the process and allows for version control.
  - Great Expectations is integrated into our pipelines to ensure comprehensive data validation checks at every stage. This helps us detect and address data quality issues early, preventing downstream errors.
  - dbt docs are used for good documentations. This allows for data lineage tracking, and helps downstream consumers discover and understand the datasets we curate for them.
- **Ad-Hoc Analysis: Flexibility and Speed**
  - Depending on dataset size and query patterns, we also leverage DuckDB for ad-hoc analysis and rapid prototyping. DuckDB's in-process, embeddable nature allows for fast, interactive querying, particularly for smaller datasets or exploratory analysis.
- **Medallion Architecture: Organizing Data for Consumption**
  - We implemented a medallion architecture (Bronze, Silver, Gold) to organize our data for optimal consumption.
  - The Bronze layer stores raw data, the Silver layer contains cleaned and conformed data, and the Gold layer provides business-ready datasets.
  - The Gold layer is further refined to create fine-grained datasets tailored to specific data access patterns. This approach minimizes data scanning during queries, significantly optimizing query performance, especially for frequently accessed data.

To enable efficient data discovery and querying:

- **Data Indexing and Metastore: Seamless Data Discovery**
  - AWS Glue crawlers automatically index data in S3, updating metadata as new data arrives.
  - The AWS Glue Data Catalog serves as our Hive Metastore, providing a centralized repository for metadata. This allows Trino to efficiently locate and access data across our data lake.
- **Querying and Visualization: Empowering Data-Driven Decisions**
  - Trino is integrated with the Hive Metastore for distributed querying, enabling us to query data across our data lake using standard SQL. Trino's ability to federate queries across multiple data sources provides flexibility.
  - Metabase is linked to Trino, providing a user-friendly data visualization layer. This empowers our data and BI teams to create interactive reports and dashboards, driving data-driven decisions throughout the organization.

## Cost Analysis

[^1]: This is the footnote.
