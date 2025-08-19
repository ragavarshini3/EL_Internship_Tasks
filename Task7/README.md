📊 Task 7 – Sales Summary (PostgreSQL + Python)
🔹 Objective

Connect Python to PostgreSQL

Run SQL query to get total quantity & revenue per product

Display results in a pandas DataFrame + bar chart

🔹 Tools

PostgreSQL

Python 3.12

Libraries: psycopg2, pandas, matplotlib

🔹 SQL Used
SELECT 
    product, 
    SUM(quantity) AS total_qty, 
    SUM(quantity * price) AS revenue
FROM sales
GROUP BY product;

🔹 Python Script
import psycopg2, pandas as pd, matplotlib.pyplot as plt

conn = psycopg2.connect(dbname="salesdb", user="postgres", password="yourpassword", host="localhost", port="5432")
df = pd.read_sql_query("""
SELECT product, SUM(quantity) AS total_qty, SUM(quantity * price) AS revenue 
FROM sales GROUP BY product
""", conn)
conn.close()

print(df)
df.plot(kind='bar', x='product', y='revenue')
plt.title("Revenue by Product")
plt.savefig("sales_chart.png")
plt.show()

🔹 Output

📋 Example Table

  product     total_qty   revenue
0  Laptop            5   306000
1   Phone            8   129000
2 Headphones        18    31000


📊 Example Chart


✅ Outcome: Learned SQL + Python integration, generated sales summary & chart.