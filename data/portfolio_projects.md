# Credit Card Offer Analysis (Rewards Optimization)

## URL
https://github.com/aleivaar94/BMO-Airmiles-Analysis

## Purpose
Evaluate whether switching from a cash-back credit card to a BMO Air Miles rewards card yields higher net value given real personal spending patterns, and provide a data-driven recommendation.

## Tech Stack
Python, Pandas, NumPy, Matplotlib, Seaborn, Jupyter Notebook.

## Technical Highlights
Implemented a reproducible pipeline that ingests personal expense data, cleans mixed-format currency and missing values, and standardizes temporal fields for time-based aggregation. Modeled tiered rewards using floor-division logic to accurately reflect how miles are awarded across standard and BMO reward schemes, and computed alternative reward scenarios (standard, BMO, BMO World Elite) including annual-fee adjustments. Aggregated spending by year, merchant (e.g., Safeway), and payment method to isolate miles-earning transactions, and used visualizations and grouped statistics to surface purchase frequency and variance that drive rewards outcomes.

## Skills Demonstrated
Data engineering and cleaning of real-world financial spreadsheets, quantitative reward-modeling and scenario analysis, exploratory data analysis and visualization, and translating analytical results into actionable financial recommendations.

## Result/Impact
The analysis found the BMO Air Miles World Elite card would have yielded approximately $300 in rewards for 2021—about $111 more than the current Scotiabank cash-back card—while 2022 results were mixed (estimated ~$167 vs ~$181), highlighting the importance of category-specific spending and temporal accumulation when choosing rewards products. This evidence-based assessment supports an informed card-switch decision and identifies high-impact spending categories (e.g., grocery purchases at Safeway) to maximize returns.

# Renew Amazon Prime? A Cost-Benefit Analysis

## URL
https://github.com/aleivaar94/Renew-Amazon-Prime-2022

## Purpose
Analyze personal Amazon order history to determine whether renewing an Amazon Prime membership is cost-effective after a 20% price increase, using historical spending patterns and modeled shipping costs.

## Tech Stack
Python, Jupyter Notebook, pandas, numpy, seaborn, matplotlib, CSV data (personal order export).

## Technical Highlights
Cleaned and normalized the raw Amazon order export, including sensitive-data removal, numeric coercion, and datetime parsing. Aggregated orders by order_id and order_date and derived year/month features to enable per-year analysis. Performed EDA with boxplots and histograms to demonstrate distributional skewness and justify median-based interpretation. Implemented a conservative shipping-cost model (assigning $7.99 for orders < $25) and aggregated yearly shipping costs to compare against the annual Prime fee, producing net savings estimates.

## Skills Demonstrated
Data ingestion and cleaning, exploratory data analysis, statistical summary interpretation, scenario modeling and assumptions, and clear visualization for stakeholder decision-making within a reproducible Jupyter workflow.

## Result/Impact
Quantified the financial trade-off: using the modeled conservative shipping rates, the analysis shows net annual savings (example: 2021 would have incurred approximately $108.74 more in shipping without Prime), supporting renewal even after a 20% membership price increase. The notebook provides a repeatable method to reassess the decision annually.

# SQL Database of Save-On-Foods products extracted using API\

## URL
https://github.com/aleivaar94/SQL-Database-of-Save-On-Foods-Products-Extracted-Using-API/blob/master/images/save-on-foods-logo.png

## Purpose
This project programmatically extracts product data from an e‑commerce API to build a clean, queryable dataset for analysis and downstream tooling. It demonstrates a repeatable ETL workflow to turn paginated JSON API results into analytics-ready CSV and relational data.

## Tech Stack
Python, Jupyter Notebook, pandas, requests, numpy, json, SQLAlchemy, SQLite, Postman (for request prototyping), and Git.

## Technical Highlights
The solution identifies the vendor API via browser developer tools, converts the API response into structured data, and implements pagination by incrementing the API skip/take parameters to reliably iterate through results. JSON responses are normalized into tabular form using pandas' json_normalize, followed by targeted cleaning and column transformations to parse price and unit fields. The cleaned dataframe is exported to CSV and persisted into a SQLite relational table with explicit column types via SQLAlchemy for easy querying and integration.

## Skills Demonstrated
Practical skills include API-driven data extraction, ETL design, JSON normalization, data cleaning and transformation, relational schema creation, and reproducible analysis in a notebook environment. It also demonstrates use of tooling for API inspection and request generation (Postman) and basic database engineering for analytics.

## Result/Impact
The pipeline extracted and consolidated 1,519 meat-related product records into a CSV and a SQLite database, enabling efficient SQL queries and downstream analysis. This reproducible workflow reduces manual scraping effort and provides a reliable foundation for price analysis, category insights, and inventory-style analytics.