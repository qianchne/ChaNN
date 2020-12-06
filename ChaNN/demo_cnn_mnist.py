### chan 2020/11/24
### build a cnn for nmist classification task based on numpy
import numpy as np
from layers import *
from loss import loss_MSE, loss_CrossEntropy
from activation import ReLU_forward, ReLU_backward
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
    #print(train_y[idx])
    return train_x[idx],train_y[idx]
#x,y= next_batch(16)
#print("x.shape:{},y.shape:{}".format(x.shape,y.shape))


### network architecture
##  input(batch, 1, 28, 28)
##  conv1(1, filters, 3, 3)(strides(1,1),padding(0,0)) -> output(batch, filters, 26, 26)
##  maxpooling(2,2)(strides(2,2),padding(0,0)) -> output(batch, filters, 13, 13)
##  flatten -> output(batch, 13*13)
##  fc1(169, 64) -> output(batch, 64)
##  fc2(64, 10) -> output(batch, 10)

### define weights, neurons, gradients
weights = {}
nuerons={}
gradients={}
weights_scale = 1e-2
filters = 1
fc_units=64


### random initiation
weights["K1"] = weights_scale * np.random.randn(1, filters, 3, 3).astype(np.float64)
weights["b1"] = np.zeros(filters).astype(np.float64)
weights["W2"] = weights_scale * np.random.randn(filters * 13 * 13, fc_units).astype(np.float64)
weights["b2"] = np.zeros(fc_units).astype(np.float64)
weights["W3"] = weights_scale * np.random.randn(fc_units, 10).astype(np.float64)
weights["b3"] = np.zeros(10).astype(np.float64)

'''
### zero initiation
weights["K1"] = weights_scale * np.zeros([1, filters, 3, 3]).astype(np.float64)
weights["b1"] = np.zeros(filters).astype(np.float64)
weights["W2"] = weights_scale * np.zeros([filters * 13 * 13, fc_units]).astype(np.float64)
weights["b2"] = np.zeros(fc_units).astype(np.float64)
weights["W3"] = weights_scale * np.zeros([fc_units, 10]).astype(np.float64)
weights["b3"] = np.zeros(10).astype(np.float64)
'''

### load pre-trained weights as initiation
#weights = np.load('weights/batch2_steps2000.npy', allow_pickle=True).item()

### forward
def forward(X):
    nuerons["conv1"]=conv_forward(X.astype(np.float64),weights["K1"],weights["b1"])
    #print(nuerons["conv1"].shape)
    nuerons["conv1_relu"]=ReLU_forward(nuerons["conv1"])
    #print(nuerons["conv1_relu"].shape)
    nuerons["maxp1"]=pooling_max_forward(nuerons["conv1_relu"].astype(np.float64),poolKernel=(2,2))
    #print(nuerons["maxp1"].shape)
    nuerons["flatten"]=flatten_forward(nuerons["maxp1"])
    #print(nuerons["flatten"].shape)
    nuerons["fc2"]=fc_forward(nuerons["flatten"],weights["W2"],weights["b2"])
    #print(nuerons["fc2"].shape)
    nuerons["fc2_relu"]=ReLU_forward(nuerons["fc2"])
    #print(nuerons["fc2_relu"].shape)
    nuerons["y"]=fc_forward(nuerons["fc2_relu"],weights["W3"],weights["b3"])
    #print(nuerons["y"].shape)
    return nuerons["y"]

### backward
def backward(X,y_true):
    #print('backward')
    loss,dy=loss_CrossEntropy(nuerons["y"],y_true)
    #print(dy.shape)
    gradients["W3"],gradients["b3"],gradients["fc2_relu"]=fc_backward(dy,weights["W3"],nuerons["fc2_relu"])
    #print(gradients["fc2_relu"].shape)
    gradients["fc2"]=ReLU_backward(gradients["fc2_relu"],nuerons["fc2"])
    #print(gradients["fc2"].shape)
    gradients["W2"],gradients["b2"],gradients["flatten"]=fc_backward(gradients["fc2"],weights["W2"],nuerons["flatten"])
    #print(gradients["flatten"].shape)
    gradients["maxp1"]=flatten_backward(gradients["flatten"],nuerons["maxp1"])
    #print(gradients["maxp1"].shape)
    gradients["conv1_relu"]=pooling_max_backward(gradients["maxp1"].astype(np.float64),nuerons["conv1_relu"].astype(np.float64),poolKernel=(2,2))
    #print(gradients["conv1_relu"].shape)
    gradients["conv1"]=ReLU_backward(gradients["conv1_relu"],nuerons["conv1"])
    #print(gradients["conv1"].shape)
    #print(weights["K1"].shape)
    #print(X.shape)
    gradients["K1"],gradients["b1"],_=conv_backward(gradients["conv1"],weights["K1"],X)
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
    #sgd=SGD(lr=0.0001,decay=1e-6)
    #adg = AdaGrad(lr=0.0001)
    sgd=SGD_momentum(lr=0.001,decay=1e-6)
    #rms = RMSProp(lr=0.001)
    #adam = Adam(lr=0.01)

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
        sgd.iterate(weights, gradients)

        ## print result
        if s % 200 == 0:
            print("\n step:{} ; loss:{}".format(s,loss))
            idx=np.random.choice(len(val_x),200)
            print(" train_acc:{};  val_acc:{}".format(_accuracy(X,y), _accuracy(val_x[idx],val_y[idx])))

            pointsX.append(s)
            pointsLoss.append(loss)
            pointsTrain_acc.append(_accuracy(X,y))
            pointsVal_acc.append(_accuracy(val_x[idx],val_y[idx]))

    #np.save('weights/batch'+str(batchSize)+'_steps'+str(steps)+'.npy', weights)
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
    plt.xlim(0, 2000)
    plt.ylim(0, 10)
    ax1.xaxis.set_major_locator(plt.MultipleLocator(200))
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
    plt.xlim(0, 2000)
    plt.ylim(0, 1)
    ax2.xaxis.set_major_locator(plt.MultipleLocator(200))
    ax2.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    plt.tight_layout()
    plt.grid(linestyle='-.')
    plt.show()


if __name__ == "__main__":
    training()
    #weights = np.load('weights/batch2_steps2000.npy', allow_pickle=True).item()
    predict()

    #pointsX = [0,200,400,600,800,1000,1200,1400,1600,1800]
    #pointsLoss = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    #pointsTrain_acc = [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.0]
    #pointsVal_acc = [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.0]
    #plotFig(pointsX, pointsLoss, pointsTrain_acc, pointsVal_acc)
