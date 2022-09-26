import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def genData():
    # Generate dataset
    n_data = 1000000
    dataset = np.zeros([n_data, 3])
    cpt = 0
    for u_iter in np.arange(-5,5,0.01):
        for v_iter in  np.arange(-5,5,0.01):
            dataset[cpt,0] = u_iter
            dataset[cpt,1] = v_iter
            dataset[cpt,2] = np.sin(u_iter-v_iter)
            cpt+=1
    # Shuffle and split dataset
    np.random.shuffle(dataset)
    train_data = dataset[:int(n_data*0.7)]
    val_data = dataset[int(n_data*0.7):int(n_data*0.85)]
    test_data = dataset[int(n_data*0.85):]
    return train_data, val_data, test_data


class Net(nn.Module):
    def __init__(self, nins, nout):
        super(Net, self).__init__()
        self.nins = nins
        self.nout = nout
        nhid = 10
        self.hidden = nn.Linear(nins, nhid)
        self.out = nn.Linear(nhid, nout)

    def forward(self, x):
        x = torch.tanh(self.hidden(x))
        x = self.out(x)
        return x


def test(model, data):
    x = torch.tensor(data[:,:-1])
    y = torch.tensor(data[:,-1])
    haty = model(x)
    return 


def train(model, train_data, max_epoch):
    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    plot_values = {"epoch": [], "loss": []}

    for epoch in range(max_epoch):
        optim.zero_grad()
        np.random.shuffle(train_data)
        x = torch.FloatTensor(train_data[:,:-1])
        y = torch.FloatTensor(train_data[:,-1])
        haty = model(x).view(700000)
        loss = criterion(haty, y)
        print(f"epoch {epoch}: loss = {loss}")
        loss.backward()
        optim.step()
        plot_values["epoch"].append(epoch)
        plot_values["loss"].append(loss.item())
    # Plot loss evolution
    _, ax = plt.subplots()
    ax.plot(plot_values["epoch"], plot_values["loss"])
    ax.set(xlabel='epoch', ylabel='MSE loss', title='MSE loss evolution with training')
    ax.axis(ymin=0)
    ax.grid()
    # fig.savefig("loss.png")
    plt.show()


def main():
    # Generate data
    train_data, val_data, test_data = genData()
    # Create model
    model = Net(2,1)
    # Train model
    train(model, train_data, 100)


if __name__ == "__main__":
    main()