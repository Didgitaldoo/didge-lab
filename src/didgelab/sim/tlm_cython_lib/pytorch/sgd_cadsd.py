import torch
from torch.nn import MSELoss
from didgelab.sim.tlm_cython_lib.pytorch.cadsd_pytorch import Segment, cadsd_Ze
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

fmin=50
fmax=100

class DidgeModel(torch.nn.Module):

    def __init__(self):
        torch.nn.Module.__init__(self)

        self.length=torch.nn.Parameter(torch.tensor([1200.0]))
        self.d0=32
        self.d0=torch.tensor([32])
        self.d1=32
        self.d1=torch.nn.Parameter(torch.tensor([80.0]))
        
    def get_segments(self):
        seg1=torch.tensor([0, self.d0])
        seg2=torch.cat((self.length, self.d1))
        segments=torch.stack((seg1, seg2))
        segments=Segment.create_segments_from_geo(segments)
        return segments

    def forward(self, fmin, fmax):
        segments=self.get_segments()
        impedances=[]
        for f in range(fmin, fmax):
            impedances.append(cadsd_Ze(segments, f))
        return torch.stack(impedances)


def plot_model(model):
    segments=model.get_segments()
    impedances=[]
    x=[]
    for f in range(fmin, fmax):
        x.append(f)
        impedances.append(cadsd_Ze(segments, f).detach())
    impedances=np.array(impedances)
    impedances=impedances/impedances.max()
    return x, impedances




def train():

    f0=73
    target_freqs=[f0]
    width=100

    x=np.arange(fmin, fmax)
    y=[]
    for _x in x:
        f=np.argmin([np.abs(f-_x) for f in target_freqs])
        f=target_freqs[f]
        d=np.abs(_x-f)
        y.append(d*-1)
    y=np.array(y)
    target_y=torch.tensor(y/y.max())
    plt.plot(x,y)
    plt.show()
    return


    model=DidgeModel()


    criterion = MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=100, momentum=0.9)

    
    for i in range(200):
        optimizer.zero_grad()

        outputs = model(fmin, fmax)
        outputs=outputs/outputs.max()

        region=np.arange(fmin, f0+width)
        region-=fmin

        # plot computational graph
        # dot = make_dot(outputs, params=dict(model.named_parameters()), show_attrs=True, show_saved=True) 
        # dot.render("test.png")

        loss = criterion(outputs, target_y)
        loss.backward()
        optimizer.step()

        if i%50==0:
            print(i, "loss", loss.item())
            print(list(model.parameters()))

            plt.clf()
            plt.plot(x, outputs.detach().numpy())
            plt.plot(x, target_y)
            plt.savefig(f"test{i}.png")

    # # for local maxima
    # x,y=plot_model(model)
    # plt.plot(x,y)

    # mini=np.array(argrelextrema(y, np.greater))+fmin
    # print(mini)


    # plt.plot(x,target_y)


    # plt.plot(x, impedances)
    # plt.show()



    # inputs, labels = data

    # # zero the parameter gradients
    # optimizer.zero_grad()

    # # forward + backward + optimize
    # outputs = net(inputs)
    # loss = criterion(outputs, labels)
    # loss.backward()
    # optimizer.step()


    


if __name__=="__main__":

    train()
