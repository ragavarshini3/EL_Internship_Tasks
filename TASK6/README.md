# EL_TASK_6
README.md — Task 6: Sales Trend Analysis
📌 Overview

This project is part of my Elevate Labs Internship – Data Analytics Track.
Task 6 focuses on analyzing sales data from an e-commerce dataset stored in PostgreSQL, and generating monthly revenue trends, top months by revenue, and year-specific breakdowns.

🗂 Dataset

File: online_sales.csv
Rows: 49,782
Columns:

InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country, Discount, PaymentMethod, ShippingCost, Category, SalesChannel, ReturnStatus, ShipmentProvider, WarehouseLocation, OrderPriority

⚙️ Steps Performed

Created PostgreSQL Table matching CSV columns.

Imported Data into PostgreSQL using \copy.

Cleaned Data by ensuring correct data types & handling nulls.

Analyzed Trends using SQL aggregation functions.

Exported Results as CSV for reporting.

📊 Queries
1️⃣ Monthly Revenue & Order Count
SELECT
    DATE_TRUNC('month', invoicedate)::DATE AS month,
    SUM(COALESCE(quantity,0) * COALESCE(unitprice,0)
        - COALESCE(discount,0)
        + COALESCE(shippingcost,0)) AS monthly_revenue,
    COUNT(DISTINCT invoiceno) AS order_count
FROM online_sales.orders
GROUP BY 1
ORDER BY 1;

2️⃣ Top 3 Months by Revenue
SELECT
    DATE_TRUNC('month', invoicedate)::DATE AS month,
    SUM(COALESCE(quantity,0) * COALESCE(unitprice,0)
        - COALESCE(discount,0)
        + COALESCE(shippingcost,0)) AS monthly_revenue
FROM online_sales.orders
GROUP BY 1
ORDER BY monthly_revenue DESC
LIMIT 3;

3️⃣ Year-Specific Breakdown (Example: 2024)
SELECT
    EXTRACT(MONTH FROM invoicedate)::INT AS month,
    SUM(COALESCE(quantity,0) * COALESCE(unitprice,0)
        - COALESCE(discount,0)
        + COALESCE(shippingcost,0)) AS monthly_revenue,
    COUNT(DISTINCT invoiceno) AS order_count
FROM online_sales.orders
WHERE EXTRACT(YEAR FROM invoicedate) = 2024
GROUP BY 1
ORDER BY 1;

📂 Output Files

monthly_sales.csv → Full monthly trend

top_3_months.csv → Top 3 months by revenue

2024_sales.csv → 2024 monthly breakdown

🛠 Tools & Technologies

Database: PostgreSQL 17

Import: \copy command (psql)

Export: \copy command (psql)

IDE/Client: pgAdmin 4, SQL Shell (psql)

📌 Key Insights

Sales peaks occur in specific months, indicating seasonal trends.

Higher revenue often correlates with higher average order values.

Year-wise filtering helps identify growth patterns.
