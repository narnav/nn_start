import numpy as np

# a = np.array([1, 2, 3])
# b = np.array([4, 5, 6])
# result = (a @ b)  # 1*4 + 2*5 + 3*6 = 32
# print(result)  # Output: 32

a = np.array([[1, 2],
              [3, 4]])
b = np.array([[5, 6],
              [7, 8]])
result = np.dot(a, b)
print(result)  # Output: [[19 22]
                                #   [43 50]]
