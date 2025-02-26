---
date: "2025-02-27T00:34:08+05:30"
title: "Building a Cloud Agnostic Data Platform"
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

![Data Platform](/images/Initial_Data_Infra.png)
Our initial data platform was built on a foundation of tools that addressed our early-stage needs, but quickly became insufficient as we scaled. Data ingestion was primarily handled by Hevo, which leveraged Debezium Slots for capturing CDC events from our databases, oplog for MongoDB, and direct integration with Google Sheets. While Hevo simplified initial data capture, its data transformation capabilities were limited, primarily offering basic functionality like data key deletion and value formatting.

Revenue data from marketplaces was ingested through an RPA-driven process, with data being directly ingested into Google BigQuery (GBQ) as raw dumps. BigQuery (GBQ). While this approach was simpler, it came with high costs, as GBQ is priced based on the amount of data queried. Given that the data sizes for each table were in the order of 200-500GBs, the costs quickly escalated.

Furthermore, a significant portion of queries were executed directly against our live OLTP tables. This direct querying increased the load on our production databases, impacting performance and further contributing to cost increases.

In the early days, with a smaller team and less data, these engineering decisions were pragmatic and likely solved the immediate problems. However, as our company grew and data demands increased, it became clear that this solution was not scalable and could not meet our evolving requirements. This realization led to the creation of a new data team, with myself and my manager, Aankesh [^1], tasked with building a more robust and scalable data platform. We needed a platform that could handle the volume, variety, and complexity of our data, while also providing the necessary tools for efficient analysis and decision-making.

## Data Platform

## Cost Analysis

[^1]: This is the footnote.
