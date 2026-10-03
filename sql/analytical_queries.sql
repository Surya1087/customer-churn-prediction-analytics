-- Overall KPIs
SELECT COUNT(*) AS customers, SUM(churn) AS churned_customers,
       ROUND(100.0 * SUM(churn) / COUNT(*), 2) AS churn_rate_pct
FROM telco_customers;

-- Contract and payment comparisons
SELECT contract, COUNT(*) customers, SUM(churn) churned_customers,
       ROUND(100.0 * SUM(churn) / COUNT(*), 2) churn_rate_pct
FROM telco_customers GROUP BY contract ORDER BY churn_rate_pct DESC;

SELECT payment_method, COUNT(*) customers, SUM(churn) churned_customers,
       ROUND(100.0 * SUM(churn) / COUNT(*), 2) churn_rate_pct
FROM telco_customers GROUP BY payment_method ORDER BY churn_rate_pct DESC;

-- CASE WHEN tenure grouping
SELECT CASE WHEN tenure_months < 6 THEN '0-5' WHEN tenure_months < 12 THEN '6-11'
 WHEN tenure_months < 24 THEN '12-23' WHEN tenure_months < 48 THEN '24-47' ELSE '48+' END tenure_group,
 COUNT(*) customers, ROUND(100.0 * SUM(churn) / COUNT(*), 2) churn_rate_pct
FROM telco_customers GROUP BY tenure_group;

-- Revenue by status
SELECT CASE churn WHEN 1 THEN 'Churned' ELSE 'Retained' END status,
 COUNT(*) customers, ROUND(AVG(monthly_charges),2) avg_monthly_charges,
 ROUND(AVG(total_charges),2) avg_total_charges
FROM telco_customers GROUP BY churn;

-- HAVING and a subquery for actionable segments
SELECT contract, payment_method, COUNT(*) customers,
 ROUND(100.0 * SUM(churn) / COUNT(*),2) churn_rate_pct
FROM telco_customers GROUP BY contract, payment_method
HAVING COUNT(*) >= 50 AND 100.0 * SUM(churn) / COUNT(*) >= 30;

SELECT customer_id, monthly_charges, contract, churn FROM telco_customers
WHERE monthly_charges > (SELECT AVG(monthly_charges) FROM telco_customers)
ORDER BY monthly_charges DESC LIMIT 100;
