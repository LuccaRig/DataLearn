import torch
import numpy as np

# Simple Data manipulation with tensors

# data = [[1, 2], [3, 4]]
# x_data = torch.tensor(data)

# np_array = np.array(data)
# x_np = torch.from_numpy(np_array)

# x_ones = torch.ones_like(x_data) # retains the properties of x_data
# print(f"Ones Tensor: \n {x_ones} \n")

# x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
# print(f"Random Tensor: \n {x_rand} \n")

# Creating 3 tensors with differents characteristics like random tensor, tensor with ones and with zeros

# shape = (2,3,)
# rand_tensor = torch.rand(shape)
# ones_tensor = torch.ones(shape)
# zeros_tensor = torch.zeros(shape)

# print(f"Random Tensor: \n {rand_tensor} \n")
# print(f"Ones Tensor: \n {ones_tensor} \n")
# print(f"Zeros Tensor: \n {zeros_tensor}")


# Index slicing

# tensor = torch.ones(4, 4)
# print(f"First row: {tensor[0]}")
# print(f"First column: {tensor[:, 0]}")
# print(f"Last column: {tensor[..., -1]}")
# tensor[:,1] = 0
# print(tensor)