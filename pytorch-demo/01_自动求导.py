import torch
from torch.autograd import Variable
import torchvision
import torch.nn as nn
import numpy as np


"""
    pip install torch==1.7.1 torchvision==0.8.2 -f https://download.pytorch.org/whl/torch_stable.html 
"""


def test0():
    # tensor 基本使用
    print(torch.tensor([1, 2]))
    print(torch.tensor(np.array([1, 2])))
    print(torch.from_numpy(np.array([1, 2])))
    t = torch.tensor([1, 2])
    print(t.shape)
    print(t.dtype)
    print(t.device)

    if torch.cuda.is_available():
        print("cuda is available")
        t = t.to('cuda')

    print(torch.cat([t, t]))
    print(t.size())


def test1():
    x = Variable(torch.randn(5, 5))
    y = Variable(torch.randn(5, 5))
    z = Variable(torch.randn(5, 5), requires_grad=True)
    a = x + y
    print(a.requires_grad)
    b = a + z
    print(b.requires_grad)


def test2():
    model = torchvision.models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    # Replace the last fully-connected layer
    # Parameters of newly constructed modules have requires_grad=True by default
    model.fc = nn.Linear(512, 100)

    # Optimize only the classifier
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)


def test3():
    """
    volatile 过期，使用 with torch.no_grad() 替代
    :return:
    """
    regular_input = Variable(torch.randn(64, 3, 7, 7))
    # volatile_input = Variable(torch.randn(64, 3, 7, 7), volatile=True)
    with torch.no_grad():
        volatile_input = Variable(torch.randn(64, 3, 7, 7), requires_grad=False)

    model = torchvision.models.resnet18(pretrained=True)
    print(model(regular_input).requires_grad)
    print(model(volatile_input).requires_grad)
    # print(model(volatile_input).volatile)
    print(dir(model(volatile_input)))
    # print(model(volatile_input).creator is None)


if __name__ == "__main__":
    test0()