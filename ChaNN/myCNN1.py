### chan 2020/12/05
### build a deeper CNN for nmist classification task based on numpy

import numpy as np
from layers import *
from loss import loss_MSE, loss_CrossEntropy
from activation import ReLU_forward, ReLU_backward, LeakyReLU_backward, LeakyReLU_forward
from optimizers import SGD, AdaGrad, SGD_momentum, RMSProp, Adam
import matplotlib.pyplot as plt

### 'data prepare' copied from [https://github.com/yizt/numpy_neural_network]
from temp_load_mnist import load_mnist_datasets
from temp_utils import to_categorical
## load data
train_set, val_set, test_set = load_mnist_datasets('mnist.pkl.gz')
train_x,val_x,test_x=np.reshape(train_set[0],(-1,1,28,28)),np.reshape(val_set[0],(-1,1,28,28)),np.reshape(test_set[0],(-1,1,28,28))
train_y,val_y,test_y=to_categorical(train_set[1]),to_categorical(val_set[1]),to_categorical(test_set[1])
## choose traning samples randomly
train_num = train_x.shape[0]
def next_batch(batch_size):
    idx=np.random.choice(train_num,batch_size)
    return train_x[idx],train_y[idx]
#x,y= next_batch(16)
#print("x.shape:{},y.shape:{}".format(x.shape,y.shape))


### define weights, neurons, gradients
weights = {}
nuerons={}
gradients={}
weights_scale = 1e-2

### load pre-trained weights as initiation
#weights = np.load('weights/myCNN1_batch2_steps2000.npy', allow_pickle=True).item()

### random initiation
## conv 1
weights["conv1_w"] = weights_scale * np.random.randn(1, 3, 3, 3).astype(np.float64)
weights["conv1_b"] = np.zeros(3).astype(np.float64)
## conv 2
weights["conv2_w"] = weights_scale * np.random.randn(3, 3, 4, 4).astype(np.float64)
weights["conv2_b"] = np.zeros(3).astype(np.float64)
## fc_1
weights["fc1_w"] = weights_scale * np.random.randn(75, 128).astype(np.float64)
weights["fc1_b"] = weights_scale * np.zeros(128).astype(np.float64)
## fc_2
weights["fc2_w"] = weights_scale * np.random.randn(128, 32).astype(np.float64)
weights["fc2_b"] = weights_scale * np.zeros(32).astype(np.float64)
## fc_3
weights["fc3_w"] = weights_scale * np.random.randn(32, 10).astype(np.float64)
weights["fc3_b"] = weights_scale * np.zeros(10).astype(np.float64)

### forward
def forward(X):
    #nuerons["conv1"]=conv_forward(X.astype(np.float64),weights["K1"],weights["b1"])
    nuerons["conv1"]=conv_forward(X.astype(np.float64), weights["conv1_w"], weights["conv1_b"])
    #print(nuerons["conv1"].shape)  (2,3,26,26)
    nuerons["conv1_prelu"]=LeakyReLU_forward(nuerons["conv1"])
    #print(nuerons["conv1_prelu"].shape)   (2,3,26,26)
    nuerons["maxp1"]=pooling_max_forward(nuerons["conv1_prelu"].astype(np.float64),poolKernel=(2,2))
    #print(nuerons["maxp1"].shape)   (2,3,13,13)
    nuerons["conv2"]=conv_forward(nuerons["maxp1"].astype(np.float64), weights["conv2_w"], weights["conv2_b"])
    #print(nuerons["conv2"].shape)   (2, 10, 10, 10)
    nuerons["conv2_prelu"]=LeakyReLU_forward(nuerons["conv2"])
    #print(nuerons["conv2_prelu"].shape)   #(2, 10, 10, 10)
    nuerons["maxp2"]=pooling_max_forward(nuerons["conv2_prelu"].astype(np.float64),poolKernel=(2,2))
    #print(nuerons["maxp2"].shape)    #(2,10,5,5)

    nuerons["flatten"]=flatten_forward(nuerons["maxp2"])
    #print(nuerons["flatten"].shape)  #(2,250)
    nuerons["fc1"]=fc_forward(nuerons["flatten"],weights["fc1_w"],weights["fc1_b"])
    #print(nuerons["fc1"].shape)
    nuerons["fc1_prelu"]=LeakyReLU_forward(nuerons["fc1"])
    #print(nuerons["fc1_prelu"].shape)
    nuerons["fc2"]=fc_forward(nuerons["fc1_prelu"],weights["fc2_w"],weights["fc2_b"])
    #print(nuerons["fc2"].shape)
    nuerons["fc2_prelu"]=LeakyReLU_forward(nuerons["fc2"])
    #print(nuerons["fc2_prelu"].shape)
    nuerons["y"]=fc_forward(nuerons["fc2_prelu"],weights["fc3_w"],weights["fc3_b"])
    #print(nuerons["y"].shape)
    return nuerons["y"]

### backward
def backward(X,y_true):
    #print('backward')
    loss,dy=loss_CrossEntropy(nuerons["y"],y_true)
    #print(dy.shape)
    gradients["fc3_w"],gradients["fc3_b"],gradients["fc2_prelu"]=fc_backward(dy,weights["fc3_w"],nuerons["fc2_prelu"])
    #print(gradients["fc2_prelu"].shape)
    gradients["fc2"]=ReLU_backward(gradients["fc2_prelu"],nuerons["fc2"])
    #print(gradients["fc2"].shape)
    gradients["fc2_w"],gradients["fc2_b"],gradients["fc1_prelu"]=fc_backward(gradients["fc2"],weights["fc2_w"],nuerons["fc1_prelu"])
    #print(gradients["fc1_prelu"].shape)
    gradients["fc1"]=ReLU_backward(gradients["fc1_prelu"],nuerons["fc1"])
    #print(gradients["fc1_prelu"].shape)
    gradients["fc1_w"],gradients["fc1_b"],gradients["flatten"]=fc_backward(gradients["fc1"],weights["fc1_w"],nuerons["flatten"])

    #print(gradients["flatten"].shape)
    gradients["maxp2"]=flatten_backward(gradients["flatten"],nuerons["maxp2"])

    gradients["conv2_prelu"]=pooling_max_backward(gradients["maxp2"].astype(np.float64),nuerons["conv2_prelu"].astype(np.float64),poolKernel=(2,2))
    gradients["conv2"]=ReLU_backward(gradients["conv2_prelu"],nuerons["conv2"])
    gradients["conv2_w"],gradients["conv2_b"], gradients["maxp1"]=conv_backward(gradients["conv2"],weights["conv2_w"], nuerons["maxp1"])
    gradients["conv1_prelu"]=pooling_max_backward(gradients["maxp1"].astype(np.float64),nuerons["conv1_prelu"].astype(np.float64),poolKernel=(2,2))
    gradients["conv1"]=ReLU_backward(gradients["conv1_prelu"],nuerons["conv1"])
    gradients["conv1_w"],gradients["conv1_b"], gradients["conv1"]=conv_backward(gradients["conv1"],weights["conv1_w"], X)
    #print(gradients["conv1"].shape)
    #print(weights["K1"].shape)
    #print(X.shape)
    #gradients["K1"],gradients["b1"],_=conv_backward(gradients["conv1"],weights["K1"],X)
    #print('check...')
    #print(gradients["K1"].shape)
    return loss

## calculate accuracy
def _accuracy(X,y_true):
    y_predict=forward(X)
    return np.mean(np.equal(np.argmax(y_predict,axis=-1),
                            np.argmax(y_true,axis=-1)))

### training
def training(batchSize=2, steps=10000):

    ### set optimizers
    #sgd=SGD(lr=0.1,decay=1e-6)
    #adg = AdaGrad(lr=0.01)
    #sgd=SGD_momentum(lr=0.001,decay=1e-6)
    #rms = RMSProp(lr=0.001)
    adam = Adam(lr=0.001)

    ### save training process
    pointsX = []
    pointsLoss = []
    pointsTrain_acc = []
    pointsVal_acc = []
    
    ### training 
    for s in range(steps):

        ## prepare data
        X,y=next_batch(batchSize)
        ## forward
        forward(X)
        ## backward
        loss=backward(X,y)
        ## update params
        adam.iterate(weights, gradients)

        ## print result
        if s % 200 == 0:
            print("\n step:{} ; loss:{}".format(s,loss))
            idx=np.random.choice(len(val_x), 50)
            print(" train_acc:{};  val_acc:{}".format(_accuracy(X,y), _accuracy(val_x[idx],val_y[idx])))

            pointsX.append(s)
            pointsLoss.append(loss)
            pointsTrain_acc.append(_accuracy(X,y))
            pointsVal_acc.append(_accuracy(val_x[idx],val_y[idx]))

    np.save('weights/myCNN1_batch'+str(batchSize)+'_steps'+str(steps)+'.npy', weights)
    #print("\n final result test_acc:{};  val_acc:{}".format(_accuracy(test_x,test_y), _accuracy(val_x,val_y)))
    
    plotFig(pointsX, pointsLoss, pointsTrain_acc, pointsVal_acc)

    print('finish')

def predict():
    idx=np.random.choice(test_x.shape[0],3)
    x,y=test_x[idx],test_y[idx]
    y_predict = forward(x)
    for i in range(3):
        plt.figure(figsize=(3,3))
        plt.imshow(np.reshape(x[i],(28,28)))
        plt.show()
        print("y_true:{},y_predict:{}".format(np.argmax(y[i]),np.argmax(y_predict[i])))

def plotFig(pointsX, pointsLoss, pointsTrain_acc, pointsVal_acc):
    fig1 = plt.figure()
    #plt.rcParams['savefig.dpi'] = 300 # pixels
    #plt.rcParams['figure.dpi'] = 300 # resolution
    ax1 = plt.subplot(1, 1, 1)
    plt.plot(pointsX, pointsLoss, color='red', label='loss', marker='v',linestyle='-', linewidth='2')
    plt.title('mnist recognition training process: loss')
    plt.legend(loc = 'upper right', fontsize=10, frameon=False) 
    ax1.set_xlabel('# of steps', fontsize=10)
    ax1.set_ylabel('loss', fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlim(0, 20000)
    plt.ylim(0, 10)
    ax1.xaxis.set_major_locator(plt.MultipleLocator(2000))
    ax1.yaxis.set_major_locator(plt.MultipleLocator(1))
    plt.tight_layout()
    plt.grid(linestyle='-.')
    plt.show()

    fig2 = plt.figure()
    #plt.rcParams['savefig.dpi'] = 300 # pixels
    #plt.rcParams['figure.dpi'] = 300 # resolution
    ax2 = plt.subplot(1, 1, 1)
    plt.plot(pointsX, pointsVal_acc, color='red', label='val_acc', marker='v',linestyle='-', linewidth='2')
    plt.title('mnist recognition training process: val_acc')
    plt.legend(loc = 'upper right', fontsize=10, frameon=False) 
    ax2.set_xlabel('# of steps', fontsize=10)
    ax2.set_ylabel('val_acc', fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlim(0, 20000)
    plt.ylim(0, 1)
    ax2.xaxis.set_major_locator(plt.MultipleLocator(2000))
    ax2.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    plt.tight_layout()
    plt.grid(linestyle='-.')
    plt.show()


if __name__ == "__main__":
    training()
    #weights = np.load('weights/myCNN1_batch2_steps2000.npy', allow_pickle=True).item()
    predict()

    #pointsX = [0,200,400,600,800,1000,1200,1400,1600,1800]
    #pointsLoss = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    #pointsTrain_acc = [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.0]
    #pointsVal_acc = [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.0]
    #plotFig(pointsX, pointsLoss, pointsTrain_acc, pointsVal_acc)
