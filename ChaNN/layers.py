### chan 2020/11/24
### Layers for building CNN based on numpy.
### including: 
### fully connected layer, conv2D layer, pooling layer(max, avg, global max, global avg), flatten layer
import numpy as np

### fully connected layer
def fc_forward(x, W, b):
    '''
    W: weights
    x: input vector
    b: bias
    '''
    return np.dot(x, W)+b

def fc_backward(L, W, x):
    '''
    L: y_predict - y_true
    delta: here, delta = L, because no activation and only one layer. L层的误差项
    W: weights
    x: input vector
    b: bias
    
    dW: dL/dW, graident of W
    db: dL/db, gradient of b
    '''
    N = x.shape[0]
    dz = np.dot(L, W.T)
    dw = np.dot(x.T, L) 
    db = np.sum(L, axis=0) 
    return dw/N, db/N, dz

### convolutional layer
def conv_forward(x, kernel, b, padding=(0,0), strides=(1,1)):
    '''
    x: input tensor, shape(batchSize, channels, height, weight)
    kernel: shape(channels, numKernel, k1, k2)
    x * kernel = (batchSize, channels, height, weight)*(channels, numKernel, k1, k2) = (bacthSize, numKernel, height, weight)
    b: bias, shape(numKernel,)
    padding: (x,y), shape(h,w) --> shape(h+2x, w+2y)
    strides: (x direction, y direction)
    
    z: shape(batchSize, numKernel, 1+(height-k1)//strides[1], 1+(weight-k2)//strides[0])
    '''
    #assert kernel.shape[0]==x.shape[1]
    #assert b.shape[0]==kernel.shape[1]
    if padding != (0,0):
        x = np.lib.pad(x, ((0,0),(0,0),(padding[0],padding[0]),(padding[1],padding[1])), 'constant', constant_values=0)
        
    ## if height/weight cant be divided by stride, then padding zero
    channels, numKernel, k1, k2 = kernel.shape
    while (x.shape[2]-k1)%strides[1] != 0 :
        #print(x.shape[2], k1)
        x = np.lib.pad(x, ((0,0),(0,0),(0,1),(0,0)), 'constant', constant_values=0)
    while (x.shape[3]-k2)%strides[0] != 0 :
        x = np.lib.pad(x, ((0,0),(0,0),(0,0),(0,1)), 'constant', constant_values=0)
    assert x.shape[1] == channels
    batchSize, channels, height, weight = x.shape
    assert (height-k1)%strides[1] == 0 
    assert (weight-k2)%strides[0] == 0 
    #print(x.shape)

    ## calculate conv
    #print(height-k1)
    z = np.zeros((batchSize, numKernel, 1+(height-k1)//strides[1], 1+(weight-k2)//strides[0]))
    #print(z.shape)
    #print(x.shape)
    #print(kernel.shape)
    for n in np.arange(batchSize):
        for k in np.arange(numKernel):
            for h in np.arange(height-k1+1)[::strides[1]]:
                for w in np.arange(weight-k2+1)[::strides[0]]:
                    #print(n,k,h,w)
                    z[n, k, h//strides[1], w//strides[0]] = np.sum(x[n, :, h:h+k1, w:w+k2] * kernel[:,k]) + b[k]
    return z

def _insertZeros(dzNext, strides):
    '''
    For the dimention match, error_next should be transfomed 
    from (batchSize, numKernel, (H-k1)/strides[1]+1, (W-k2)/strides[0]+1  
    to (batchSize, numKernel, H, W)
    
    here, this function insert 0 in dzNext, each row and col
    for example, in dim 2 and 3, [[1,1],[2,2]] --> [[1, 0, 1],[0,0,0],[2,0,2]]
    '''
    _, _, H, W = dzNext.shape
    if strides[1] > 1:
        for row in np.arange(H-1, 0, -1):
            for n in np.arange(strides[1]-1):
                dzNext = np.insert(dzNext, row, 0, axis = 2)
    if strides[0] > 1:
        for col in np.arange(W-1, 0, -1):
            for n in np.arange(strides[1]-1):
                dzNext = np.insert(dzNext, col, 0, axis = 3)
    return dzNext

def conv_backward(dzNext, kernel, x, padding=(0,0), strides=(1,1)):
    '''
    dzNext: error term of current layer, defined as dLoss(y,y*)/dz, where z is the output of current layer
    x: input of current layer, shape(batchSize, channels, height, weight)
    kernel: convolution kernels current layer, shape(batchSize, numKernel, k1, k2)

    dK: d(dzNext)/d(weights in kernel)
    db: d(dzNext)/d(bias in kernel)
    dz: error term of layer before dzNext
    '''
    
    assert kernel.shape[1] == dzNext.shape[1]    # because number of kernel equal to channel of output feature map
    channels, numKernel, k1, k2 = kernel.shape
    batchSize, numKernel, NH, NW = dzNext.shape  # NH = (H-k1)//strides[1] +1, NW too.
    
    ### calculate dz of current layer using dzNext, according to chain rule
    ## dzNext: (batchSize, numKernel, H, W) -->(batchSize, numKernel, H, W)
    dzNextTrans1 = _insertZeros(dzNext, strides=(strides))
    dzNextTrans2 = np.lib.pad(dzNextTrans1, 
                              ((0,0),(0,0),(k1-1,k1-1),(k2-1,k2-1)), 'constant', constant_values=0) # shape(batchSize, numKernel, H+1, W+1)
    
    ## kernel: (channels, numKernel, k1, k2) --> (numKernel, channels, k1, k2) and rot_180(kernel)
    kernelTrans = np.flip(kernel, (2,3))
    kernelTrans = np.swapaxes(kernelTrans, 0, 1)
    
    ## (1) set param b to zero.    (2) default strides is (1,1) and default padding is (0,0), so dz will be (batchSize, channels, H+1 -1, W+1 -1)
    #print(dzNextTrans2.shape, kernelTrans.shape)
    dz = conv_forward(dzNextTrans2.astype(np.float64), kernelTrans.astype(np.float64), np.zeros((numKernel,), dtype=np.float64))
    
    ### remove padding
    if padding[0] > 0 and padding[1] > 0:
        dz = dz[:, :, padding[0]:-padding[0], padding[1]:-padding[1]]
    elif padding[0]>0:
        dz = dz[:, :, padding[0]:-padding[0], :]
    elif padding[1]>0:
        dz = dz[:, :, :, padding[1]:-padding[1]]
    
    
    ### calculate dk,  shape(channels, numKernel, k1, k2)
    xTrans = np.swapaxes(x, 0, 1)
    xTrans = np.lib.pad(xTrans,
                       ((0,0),(0,0),(padding[0],padding[0]),(padding[1],padding[1])), 'constant', constant_values=0)
    #print(xTrans.shape)
    #print(dzNextTrans1.shape)
    dk = conv_forward(xTrans.astype(np.float64), dzNextTrans1.astype(np.float64), np.zeros((numKernel,), dtype=np.float64))
    
    ### calculate db
    db = np.sum(np.sum(np.sum(dzNext, axis=-1), axis=-1), axis=0)

    return dk / batchSize, db / batchSize, dz

### pooling layers
## maximun pooling
def pooling_max_forward(x, poolKernel, strides=(2,2), padding=(0,0)):
    '''
    x: input feature map with shape(batchSize, channels, heights, weights)
    poolingKernel: pooling kernel
    
    z: output feature map after maximun pooling
    '''
    
    batchSize, channels, H, W = x.shape
    if padding != (0,0):
        x = np.lib.pad(x, ((0,0),(0,0),(padding[0],padding[0]),(padding[1],padding[1])), 'constant', constant_values=0)
    
    z_H = (H + 2*padding[0] - poolKernel[0]) // strides[0] + 1
    z_W = (W + 2*padding[0] - poolKernel[0]) // strides[0] + 1
    
    z = np.zeros((batchSize, channels, z_H, z_W))
    
    for n in np.arange(batchSize):
        for c in np.arange(channels):
            for h in np.arange(z_H):
                for w in np.arange(z_W):
                    z[n, c, h, w] = np.max(x[n, c, strides[0]*h : strides[0]*h+poolKernel[0], strides[1]*w : strides[1]*w+poolKernel[1]])
    
    return z

def pooling_max_backward(dzNext, x, poolKernel, strides=(2,2), padding=(0,0)):
    '''
    池化的反向传播，只是梯度的重新分配。比较容易理解，池化一般是不会有填零这一操作的。
    dzNext: error term of current layer, defined as dLoss(y,y*)/dz, where z is the output of current layer
    x: input feature map with shape(batchSize, channels, heights, weights)
    '''
    
    batchSize, channels, H, W = x.shape
    _, _, dzNext_H, dzNext_W = dzNext.shape
    
    ### padding zeros
    x = np.lib.pad(x, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant', constant_values=0)
    dz = np.zeros_like(x)
    
    ### rearrange dzNext to dz, according to the position of maximum
    for n in np.arange(batchSize):
        for c in np.arange(channels):
            for h in np.arange(dzNext_H):
                for w in np.arange(dzNext_W):
                    flatten_idx = np.argmax(x[n, c,
                                        strides[0]*h:strides[0]*h+poolKernel[0],
                                        strides[1]*w:strides[1]*w+poolKernel[1]])
                    h_idx = strides[0]*h + flatten_idx // poolKernel[1]
                    w_idx = strides[1]*w + flatten_idx % poolKernel[1]
                    dz[n, c, h_idx, w_idx] += dzNext[n, c, h, w]
    ### remove padding
    if padding[0] > 0 and padding[1] > 0:
        dz = dz[:, :, padding[0]:-padding[0], padding[1]:-padding[1]]
    elif padding[0]>0:
        dz = dz[:, :, padding[0]:-padding[0], :]
    elif padding[1]>0:
        dz = dz[:, :, :, padding[1]:-padding[1]]
    return dz

## average pooling
def pooling_average_forward(x, poolKernel, strides=(2,2), padding=(0,0)):
    '''
    x: input feature map with shape(batchSize, channels, heights, weights)
    poolingKernel: pooling kernel
    
    z: output feature map after average pooling
    '''
    
    batchSize, channels, H, W = x.shape
    if padding != (0,0):
        x = np.lib.pad(x, ((0,0),(0,0),(padding[0],padding[0]),(padding[1],padding[1])), 'constant', constant_values=0)
    
    z_H = (H + 2*padding[0] - poolKernel[0]) // strides[0] + 1
    z_W = (W + 2*padding[0] - poolKernel[0]) // strides[0] + 1
    
    z = np.zeros((batchSize, channels, z_H, z_W))
    
    for n in np.arange(batchSize):
        for c in np.arange(channels):
            for h in np.arange(z_H):
                for w in np.arange(z_W):
                    z[n, c, h, w] = np.mean(x[n, c, strides[0]*h:strides[0]*h+poolKernel[0], strides[1]*w:strides[1]*w+poolKernel[1]])
    
    return z

def pooling_average_backward(dzNext, x, poolKernel, strides=(2,2), padding=(0,0)):
    '''
    the backpropagation of pooling is the rearranement of dzNext, it has no paraments to update.
    And, there is no padding in pooling usually.
    dzNext: error term of current layer, defined as dLoss(y,y*)/dz, where z is the output of current layer
    x: input feature map with shape(batchSize, channels, heights, weights)
    '''
    
    batchSize, channels, H, W = x.shape
    _, _, dzNext_H, dzNext_W = dzNext.shape
    
    ### padding zeros
    x = np.lib.pad(x, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant', constant_values=0)
    dz = np.zeros_like(x)
    
    ### rearrange dzNext to dz
    for n in np.arange(batchSize):
        for c in np.arange(channels):
            for h in np.arange(dzNext_H):
                for w in np.arange(dzNext_W):
                    dz[n, c, 
                    strides[0]*h:strides[0]*h+poolKernel[0], 
                    strides[1]*w:strides[1]*w+poolKernel[1]] += dzNext[n, c, h, w] / (poolKernel[0]*poolKernel[1])
                    
    ### remove padding
    if padding[0] > 0 and padding[1] > 0:
        dz = dz[:, :, padding[0]:-padding[0], padding[1]:-padding[1]]
    elif padding[0]>0:
        dz = dz[:, :, padding[0]:-padding[0], :]
    elif padding[1]>0:
        dz = dz[:, :, :, padding[1]:-padding[1]]
        
    return dz

## global average pooling
def pooling_globalAverage_forward(x):
    '''
    global pooling output a value each feature map
    x: input feature map with shape(batchSize, channels, heights, weights)
    z: output feature map wih shape(batchSize, channels)
    '''
    z = np.mean(np.mean(x, axis=-1), axis=-1)
    return z

def pooling_globalAverage_backward(dzNext, dz, x):
    '''
    dzNext: error term of current layer, defined as dLoss(y,y*)/dz, where z is the output of current layer
    x: input feature map with shape(batchSize, channels, heights, weights)
    
    dz: error term of layer before dzNext
    '''
    batchSize, channels, H, W = x.shape
    
    dz = np.zeros_like(x)
    
    ### rearrange dzNext to dz
    for n in np.arange(batchSize):
        for c in np.arange(channels):
            dz[n, c, :, :] = dzNext[n,c] / H*W
    return dz

## global maximum pooling
def pooling_globalMax_forward(x):
    '''
    global pooling output a value each feature map
    x: input feature map with shape(batchSize, channels, heights, weights)
    z: output feature map wih shape(batchSize, channels)
    '''
    z = np.max(np.max(x , axis=-1), axis=-1)

def pooling_globalMax_backward(dzNext, dz, x):
    '''
    x: input feature map with shape(batchSize, channels, heights, weights)
    '''
    batchSize, channels, H, W = x.shape
    dz = np.zeros_like(x)
    
    ### rearrange dzNext to dz
    for n in np.arange(batchSize):
        for c in np.arange(channels):
            flatten_idx = np.argmax(x[n, c, :, :])
            H_idx = flatten_idx // H
            W_idx = flatten_idx % W
            dz[n, c, H_idx, W_idx] = dzNext[n,c]
    return dz

### flatten layer
def flatten_forward(x):
    N = x.shape[0]
    return np.reshape(x,(N,-1))

def flatten_backward(dzNext, x):
    return np.reshape(dzNext, np.shape(x))