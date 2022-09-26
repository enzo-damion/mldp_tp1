import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib as mpl


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


def test(model, test_data):
    x = torch.FloatTensor(test_data[:,:-1])
    y = torch.FloatTensor(test_data[:,-1])
    haty = model(x).view(150000)
    criterion = nn.MSELoss()
    loss = criterion(haty, y)

    # Prepare plot
    y = y.detach().numpy()
    haty = haty.detach().numpy()
    mpl.rcParams['agg.path.chunksize'] = 10000

    # Plot regression line
    _, ax = plt.subplots()
    ax.plot(y, haty, ".b", markersize=0.1)
    b, a = np.polyfit(y, haty, deg=1)
    xseq = np.linspace(-1, 1, num=100)
    ax.plot(xseq, a + b * xseq, color="k")
    ax.set(xlabel='Gold labels', ylabel='Prediction', title='Regression line')
    ax.grid()
    plt.show()

    # Plot 3d graph
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x[:5000,0],x[:5000,1],haty[:5000], ".b", markersize=0.5)
    plt.show()
    return loss


def train(model, train_data, max_epoch):
    optim = torch.optim.Adam(model.parameters(), lr=0.05)
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

    # Create and load model
    model = Net(2,1)
    model.load_state_dict(torch.load('./model2.pth'))

    # Train and save model
    # train(model, train_data, 2000)
    # torch.save(model.state_dict(), "./model2.pth")

    # Test model
    loss = test(model, test_data)
    print(f"test loss = {loss}")


if __name__ == "__main__":
    main()