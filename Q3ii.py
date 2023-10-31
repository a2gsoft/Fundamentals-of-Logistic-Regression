import numpy as np

def compute_p(wi, b, xi):
  """Computes the probability p of the item being in the particular class.

  Args:
    wi: A list of weights.
    b: The bias.
    xi: A list of features.

  Returns:
    The probability p of the item being in the particular class.
  """

  logit = np.sum(wi * xi) + b
  p = sigmoid(logit)
  return p

# Example usage:

wi = [0.1, 0.2, 0.3]
b = 0.4
xi = [0.5, 0.6, 0.7]

p = compute_p(wi, b, xi)

print(p)