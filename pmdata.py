import pandas as pd

# Load raw data
df = pd.read_csv("data.csv", encoding="ISO-8859-1")

# Drop missing Customer IDs
df = df.dropna(subset=['CustomerID'])

# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')

# Create Month-Year column
df['Month'] = df['InvoiceDate'].dt.to_period('M').astype(str)

# Create Revenue column
df['Revenue'] = df['Quantity'] * df['UnitPrice']

# Monthly Metrics: Revenue, Active Customers, ARPU 
monthly = (
    df.groupby("Month")
      .agg(Revenue_Monthly=("Revenue", "sum"),
           ActiveCustomers=("CustomerID", "nunique"))
      .reset_index()
)
monthly["ARPU"] = (monthly["Revenue_Monthly"] / monthly["ActiveCustomers"]).round(2)

# Merge back into main df
df = df.merge(monthly, on="Month", how="left")

# Churn Flag: inactive > 90 days 
last_purchase = df.groupby("CustomerID")["InvoiceDate"].max().reset_index()
last_purchase["DaysSinceLastPurchase"] = (df["InvoiceDate"].max() - last_purchase["InvoiceDate"]).dt.days
last_purchase["ChurnFlag"] = last_purchase["DaysSinceLastPurchase"].apply(
    lambda x: "Churned (>90d)" if x > 90 else "Active"
)
df = df.merge(last_purchase[["CustomerID", "ChurnFlag"]], on="CustomerID", how="left")

#Pareto: cumulative revenue 
cust_rev = (
    df.groupby("CustomerID")["Revenue"]
      .sum()
      .reset_index()
      .rename(columns={"Revenue": "TotalRevenue"})
      .sort_values("TotalRevenue", ascending=False)
)
cust_rev["Rank"] = range(1, len(cust_rev) + 1)
cust_rev["CumulativeRevenue"] = cust_rev["TotalRevenue"].cumsum()
cust_rev["CumulativeRevenuePct"] = (cust_rev["CumulativeRevenue"] / cust_rev["TotalRevenue"].sum()).round(4)
cust_rev["Top20PercentFlag"] = cust_rev["Rank"] <= int(0.2 * len(cust_rev))

# Merge Pareto info back to main df
df = df.merge(cust_rev[["CustomerID", "TotalRevenue", "CumulativeRevenuePct", "Top20PercentFlag"]],
              on="CustomerID", how="left")

# Save final cleaned file
df.to_csv("data_cleaned1.csv", index=False, encoding="utf-8")

print("data_cleaned1.csv created with Revenue, ARPU, ActiveCustomers, ChurnFlag, and Pareto info")
print(df.head())
