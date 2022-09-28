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


def test(model, test_data, graph=False):
    x = torch.FloatTensor(test_data[:,:-1])
    y = torch.FloatTensor(test_data[:,-1])
    haty = model(x).view(x.size()[0])
    criterion = nn.MSELoss()
    loss = criterion(haty, y)
    r2_score = 1- torch.sum((y-haty)**2) / torch.sum((y-y.float().mean())**2)
    if graph:
        # Prepare plot
        y = y.detach().numpy()
        haty = haty.detach().numpy()
        mpl.rcParams['agg.path.chunksize'] = 10000 # Increase matplotlib max render

        # Plot regression line
        _, ax = plt.subplots()
        ax.plot(y, haty, ".b", markersize=0.1)
        b, a = np.polyfit(y, haty, deg=1)  # Compute the coef of the regression line
        xseq = np.linspace(-1, 1, num=100)
        ax.plot(xseq, a + b * xseq, color="k")
        ax.set(xlabel='Gold labels', ylabel='Prediction', title="Regression line\nloss = {:.7f} & score = {:.4f}".format(loss, r2_score))
        ax.grid()
        plt.show()

        # Plot 3d graph
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        n_point = min(x.size()[0], 5000) # Number of points plotted
        ax.plot(x[:n_point,0],x[:n_point,1],haty[:n_point], ".b", markersize=0.5)
        plt.show()
    return loss, r2_score


def train(model, train_data, val_data, max_epoch):
    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    plot_values = {"epoch": [], "loss": [], "val_loss":[], "val_score":[]}

    for epoch in range(max_epoch):
        # Optimization
        optim.zero_grad()
        np.random.shuffle(train_data)
        x = torch.FloatTensor(train_data[:,:-1])
        y = torch.FloatTensor(train_data[:,-1])
        haty = model(x).view(x.size()[0])
        loss = criterion(haty, y)
        loss.backward()
        optim.step()

        # Training monitoring
        val_loss, val_score = test(model,val_data)
        plot_values["epoch"].append(epoch)
        plot_values["loss"].append(loss.item())
        plot_values["val_loss"].append(val_loss.item())
        plot_values["val_score"].append(val_score.item())
        print(f"epoch {epoch}: loss = {loss:.7f}, val_loss = {val_loss:.7f}, val_score = {val_score:.4f}")

    # Plot loss evolution
    _, ax = plt.subplots()
    ax.plot(plot_values["epoch"], plot_values["loss"], 'b', label='training loss')
    ax.plot(plot_values["epoch"], plot_values["val_loss"], 'g', label='validation loss')
    ax.set(xlabel='epoch', ylabel='MSE loss', title='Training supervision')
    ax.axis(ymin=0)
    ax.grid()
    ax.legend()
    ax2 = ax.twinx()
    ax2.plot(plot_values["val_score"], 'k', label='r2 score')
    ax2.set_ylabel("R2 score")
    ax2.axis(ymax=1)
    plt.show()


def main():
    # Generate data
    train_data, val_data, test_data = genData()

    # Create and load model
    model = Net(2,1)
    model.load_state_dict(torch.load('./model2.pth'))

    # Train and save model
    # train(model, train_data, val_data, 1000)
    # torch.save(model.state_dict(), "./model_2hidlayer.pth")

    # Test model
    test(model, test_data, graph=True)


if __name__ == "__main__":
    main()