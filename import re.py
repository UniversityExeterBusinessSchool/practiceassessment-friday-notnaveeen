import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Question 1 - Data Processing and Loops
customer_reviews = """The product is well-designed and user-friendly. However, I experienced some issues with durability. 
The customer service was helpful, but I expected a faster response. The quality of the materials used is excellent. 
Overall, the purchase was satisfactory."""

keywords = {
    0: 'user-friendly',
    1: 'helpful',
    2: 'durability',
    3: 'response',
    4: 'satisfactory',
    5: 'quality',
    6: 'service',
    7: 'issues',
    8: 'purchase',
    9: 'materials'
}

sid = "750018079"  # Given SID
first_digit = int(sid[0])  # 7
last_digit = int(sid[-1])  # 9
allocated_keywords = [keywords[first_digit], keywords[last_digit]]

keyword_counts = {key: len(re.findall(key, customer_reviews, re.IGNORECASE)) 
for key in allocated_keywords}
print("Keyword Counts:", keyword_counts)  

# Question 2 - Business Metrics
def gross_profit_margin(revenue, cogs):
    return ((revenue - cogs) / revenue) * 100

def inventory_turnover(cogs, avg_inventory):
    return cogs / avg_inventory

def customer_retention_rate(initial_customers, final_customers, new_customers):
    return ((final_customers - new_customers) / initial_customers) * 100

def break_even_analysis(fixed_costs, price_per_unit, variable_cost_per_unit):
    return fixed_costs / (price_per_unit - variable_cost_per_unit)

first_two_digits = int(sid[:2])  # 75
last_two_digits = int(sid[-2:])  # 79

gpm = gross_profit_margin(1000, first_two_digits)
it = inventory_turnover(500, last_two_digits)
crr = customer_retention_rate(1000, 800, 150)
be = break_even_analysis(5000, 50, first_two_digits)

print("Gross Profit Margin:", gpm)  
print("Inventory Turnover:", it)  
print("Customer Retention Rate:", crr)  
print("Break-even Point:", be)  
# Question 3 - Forecasting and Regression
delivery_cost = np.array([25, 30, 35, 40, 45, 50, 55, 60, 65, 70]).reshape(-1, 1)
shipment_volume = np.array([500, 480, 450, 420, 400, 370, 340, 310, 290, 250])

model = LinearRegression()
model.fit(delivery_cost, shipment_volume)

optimal_cost = -model.coef_[0] / (2 * model.intercept_)
expected_volume = model.predict(np.array([[68]]))[0]

print("Optimal Delivery Cost:", optimal_cost)
print("Expected Shipment Volume at £68:", expected_volume)

# Question 4 - Debugging and Data Visualization
import random

student_id = input("Enter your Student ID: ")
max_value = int(student_id)
random_numbers = [random.randint(1, max_value) for _ in range(100)]

plt.hist(random_numbers, bins=10, edgecolor='blue', alpha=0.7, color='red')
plt.title("Histogram of 100 Random Numbers")
plt.xlabel("Value Range")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

