import torch

'''
Matrix Operations
'''

# Uninitialized
x = torch.empty(5,3)
print (x)

# Random (vals between 0,1)
x = torch.rand(5,3)
print (x)

# Zeros
x = torch.zeros(5,3,dtype=torch.long)
print (x)

# From Data
x = torch.tensor([5.5, 3])
print (x)

# Get size (() => tuple)
print(x.size())

# Add
y = torch.rand(5,3)
print (x + y)
print (torch.add(x,y))
y.add_(x)






