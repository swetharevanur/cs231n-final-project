# WORK IN PROGRESS

class Flatten(nn.Module):
	def flatten(x):
		N = x.shape[0] # read in N, C, H, W
		return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


hidden_layer_size = 4000
learning_rate = 1e-2
input_dim = 20
num_channels = 3 

model = nn.Sequential(
	Flatten(),
	nn.Linear(input_dim * input_dim * num_channels, hidden_layer_size),
	nn.ReLU(),
	nn.Linear(hidden_layer_size, 10)
)

# you can use Nesterov momentum in optim.SGD
optimizer = optim.SGD(model.parameters(), lr=learning_rate,
					 momentum=0.9, nesterov=True)

# loader_train = DataLoader(cifar10_train, batch_size=64, 
# 					sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

def train(model, optimizer, epochs=1):
	model = model.to(device=device) # move the model parameters to CPU/GPU

	for e in range(epochs):
		for t, (x, y) in enumerate(loader_train):
			model.train()  # put model to training mode
			x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
			y = y.to(device=device, dtype=torch.long)

			scores = model(x)
			loss = F.cross_entropy(scores, y)

			optimizer.zero_grad()

			loss.backward()

			optimizer.step()

			if t % print_every == 0:
				print('Iteration %d, loss = %.4f' % (t, loss.item()))
				check_accuracy_part34(loader_val, model)
				print()


train(model, optimizer)