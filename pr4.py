# Gradient Descent Algorithm

# 1. Define the function
def function(x):
    return (x + 3) ** 2

# 2. Define the derivative of the function
def derivative(x):
    return 2 * (x + 3)

# 3. Initialize parameters
x = 2
learning_rate = 0.1
iterations = 50

print("Gradient Descent Algorithm")
print("--------------------------")
print("Initial x =", x)
print("Learning Rate =", learning_rate)

# 4. Perform Gradient Descent
for i in range(iterations):

    # Calculate gradient
    gradient = derivative(x)

    # Update x
    x = x - learning_rate * gradient

    # Calculate function value
    y = function(x)

    print(f"Iteration {i+1}: x = {x:.4f}, y = {y:.4f}")

# 5. Display minimum value
print("\n--- Final Result ---")
print("Local Minimum at x =", round(x, 4))
print("Minimum Value of y =", round(function(x), 4))
