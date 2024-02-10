import torch

state = 1.
t = torch.tensor(state, requires_grad=True)

activation = t ** 3

action = activation ** 2

optimizer = torch.optim.Adam([t])
optimizer.zero_grad()
action.backward(retain_graph=True)

optimizer.step()

action.backward(retain_graph=True)
print(t.grad)
