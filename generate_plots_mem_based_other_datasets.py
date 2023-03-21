import os
import matplotlib.pyplot as plt

# Define the data
drop = [25.52, 17.35, 14.06, 11.86, 9.44, 10.78]
quant = ['non-quantized', 'non-quantized', 'non-quantized', 'quantized', 'quantized', 'quantized']
dataset = ['CIFAR10', 'FashionMNIST', 'MNIST', 'CIFAR10', 'FashionMNIST', 'MNIST']

# Split the data by dataset
cifar10_drop = [drop[i] for i in range(len(drop)) if dataset[i] == 'CIFAR10']
fashionmnist_drop = [drop[i] for i in range(len(drop)) if dataset[i] == 'FashionMNIST']
mnist_drop = [drop[i] for i in range(len(drop)) if dataset[i] == 'MNIST']

# Create the subplots
fig, axs = plt.subplots(1, 2,  figsize=(8, 5))

# Plot the bars
axs[0].bar(['CIFAR10', 'FashionMNIST', 'MNIST'], [cifar10_drop[0], fashionmnist_drop[0], mnist_drop[0]], color='b', width=0.75)
axs[0].set_title('Non-quantized')
# axs[0].set_xlabel('Dataset')
axs[0].set_ylabel('Accuracy drop')
axs[0].set_yticklabels([f'{i}%' for i in axs[0].get_yticks()])

axs[1].bar(['CIFAR10', 'FashionMNIST', 'MNIST'], [cifar10_drop[1], fashionmnist_drop[1], mnist_drop[1]], color='b', width=0.75)
axs[1].set_title('Quantized')
# axs[1].set_xlabel('Dataset')
axs[1].set_ylabel('Accuracy drop')
axs[1].set_yticklabels([f'{i}%' for i in axs[1].get_yticks()])
axs[1].yaxis.set_label_position("right")
axs[1].yaxis.set_ticks_position('right')

# Add a common title for the figure
# fig.suptitle('Comparison of drop for CIFAR10, FashionMNIST and MNIST')

dir_name = 'graphs'
graph_name = 'histogram.pdf'
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

plt.savefig(os.path.join(dir_name, graph_name), bbox_inches='tight',dpi=300)

# Show the plot
# plt.show()
