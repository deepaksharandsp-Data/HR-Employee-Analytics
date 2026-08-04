import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# First 5 rows
print("First 5 Rows:")
print(df.head())

# Shape
print("\nDataset Shape:")
print(df.shape)

# Column names
print("\nColumn Names:")
print(df.columns)

# Dataset information
print("\nDataset Information:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nData Types:")
print(df.dtypes)
print("\nSummary Statistics:")
print(df.describe())
# Employees by Department
department_counts = df["Department"].value_counts()

department_counts.plot(kind="bar")

plt.title("Employees by Department")
plt.xlabel("Department")
plt.ylabel("Number of Employees")

plt.tight_layout()
plt.savefig("output/employees_by_department.png")
plt.show()
# Employee Attrition Count
attrition = df["Attrition"].value_counts()

print(attrition)

attrition.plot(kind="bar")

plt.title("Employee Attrition")
plt.xlabel("Attrition")
plt.ylabel("Number of Employees")

plt.tight_layout()
plt.savefig("output/employee_attrition.png")
plt.show()
# Overtime Analysis
overtime = pd.crosstab(df["OverTime"], df["Attrition"])

print(overtime)

overtime.plot(kind="bar")

plt.title("Overtime vs Attrition")
plt.xlabel("OverTime")
plt.ylabel("Number of Employees")

plt.tight_layout()
plt.savefig("output/overtime_vs_attrition.png")
plt.show()
# Average Monthly Income by Department
salary = df.groupby("Department")["MonthlyIncome"].mean()

print(salary)

salary.plot(kind="bar")

plt.title("Average Monthly Income by Department")
plt.xlabel("Department")
plt.ylabel("Average Monthly Income")

plt.tight_layout()
plt.savefig("output/average_salary_department.png")
plt.show()
# Age Distribution
plt.figure(figsize=(8,5))

plt.hist(df["Age"], bins=10)

plt.title("Employee Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Employees")

plt.tight_layout()
plt.savefig("output/age_distribution.png")
plt.show()
# Employees by Job Role
job_roles = df["JobRole"].value_counts()

print(job_roles)

job_roles.plot(kind="bar", figsize=(10,5))

plt.title("Employees by Job Role")
plt.xlabel("Job Role")
plt.ylabel("Number of Employees")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("output/job_role_distribution.png")
plt.show()
# Gender Distribution
gender = df["Gender"].value_counts()

print(gender)

gender.plot(kind="pie", autopct="%1.1f%%", startangle=90)

plt.title("Gender Distribution")
plt.ylabel("")

plt.tight_layout()
plt.savefig("output/gender_distribution.png")
plt.show()
# Correlation Matrix
correlation = df.select_dtypes(include="number").corr()

print(correlation)
plt.figure(figsize=(8,5))

plt.hist(df["MonthlyIncome"], bins=15)

plt.title("Monthly Income Distribution")
plt.xlabel("Monthly Income")
plt.ylabel("Number of Employees")

plt.tight_layout()
plt.savefig("output/monthly_income_distribution.png")
plt.show()
# Attrition by Department
attrition_dept = pd.crosstab(df["Department"], df["Attrition"])

print(attrition_dept)

attrition_dept.plot(kind="bar", figsize=(8,5))

plt.title("Attrition by Department")
plt.xlabel("Department")
plt.ylabel("Number of Employees")

plt.tight_layout()
plt.savefig("output/attrition_by_department.png")
plt.show()
# Top 10 Highest Paying Job Roles
salary_by_role = df.groupby("JobRole")["MonthlyIncome"].mean().sort_values(ascending=False)

print(salary_by_role)

salary_by_role.plot(kind="bar", figsize=(10,5))

plt.title("Average Monthly Income by Job Role")
plt.xlabel("Job Role")
plt.ylabel("Average Monthly Income")

plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("output/highest_paying_job_roles.png")
plt.show()