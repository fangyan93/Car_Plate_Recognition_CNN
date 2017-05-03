import numpy as np
import random
import tensorflow as tf
import time
import pickle
import matplotlib.pyplot as plt

# import train
def forward(W,W_pools,strd,siz_bat,num_cls,X,Y,Y_array):    
    # input layer -> convolutional layer 1
    tmp = tf.nn.conv2d(input = X, filter = W[0], strides = strd, padding = "VALID")
    tmp = tf.nn.bias_add(tmp, W[1])
    tmp = tf.nn.relu(tmp)
    tmp = tf.nn.max_pool(value = tmp, ksize = W_pools, padding = "VALID", strides = W_pools)
    print(tmp)
    # convolutional layers 1 -> convolutional layer 2
    tmp = tf.nn.conv2d(input = tmp, filter = W[2], strides = strd, padding = "VALID")
    tmp = tf.nn.bias_add(tmp, W[3])
    tmp = tf.nn.relu(tmp)
    tmp = tf.nn.max_pool(value = tmp, ksize = W_pools, padding = "VALID", strides = W_pools)
    print(tmp)
    #convert matrix to list
    tmp = tf.reshape(tmp, [siz_bat, -1])
    print(tmp)
    tmp = tf.add(tf.matmul(tmp, W[4]), W[5])
    #softmax 
    tmp = tf.nn.softmax(tmp)
    print(tmp)
    y_hat = tf.reshape(tmp, [siz_bat,num_cls])
    print(y_hat)
    loss = y_hat - Y_array
    loss = tf.multiply(loss,loss)
    loss = 0.5*tf.reduce_sum(loss)
    loss = loss/siz_bat + 0.0000001*(tf.nn.l2_loss(W[0]) + tf.nn.l2_loss(W[2]) + tf.nn.l2_loss(W[4]))

    acc = tf.argmax(y_hat, 1)
    acc = tf.to_float(acc)
    acc = tf.equal(acc, Y)
    acc = tf.to_float(acc)
    acc = tf.reduce_sum(acc)/siz_bat
    err = 1 - acc
    return([loss,err])


def Test(W,W_pools,strd,siz_bat,num_cls,X):    
    # input layer -> convolutional layer 1
    tmp = tf.nn.conv2d(input = X, filter = W[0], strides = strd, padding = "VALID")
    tmp = tf.nn.bias_add(tmp, W[1])
    tmp = tf.nn.relu(tmp)
    tmp = tf.nn.max_pool(value = tmp, ksize = W_pools, padding = "VALID", strides = W_pools)
    # convolutional layers 1 -> convolutional layer 2
    tmp = tf.nn.conv2d(input = tmp, filter = W[2], strides = strd, padding = "VALID")
    tmp = tf.nn.bias_add(tmp, W[3])
    tmp = tf.nn.relu(tmp)
    tmp = tf.nn.max_pool(value = tmp, ksize = W_pools, padding = "VALID", strides = W_pools)
    # convolutional layer 2 -> convolutional layer 3
    # tmp = tf.nn.conv2d(input = tmp, filter = W[4], strides = strd, padding = "VALID")
    # tmp = tf.nn.bias_add(tmp, W[5])
    # tmp = tf.nn.relu(tmp)
    #convert matrix to list
    tmp = tf.reshape(tmp, [siz_bat, -1])
    tmp = tf.add(tf.matmul(tmp, W[4]), W[5])
    #softmax 
    tmp = tf.nn.softmax(tmp)
    
    y_hat = tf.reshape(tmp, [siz_bat,num_cls])
    y_hat = tf.argmax(y_hat, 1)
    y_hat = tf.to_float(y_hat)
    return y_hat

#-------------------------------Loading Data-------------------------------------#
print("Loading data...")
train_label = np.loadtxt("train_label_real.txt")
train_data = np.loadtxt("train_data_real.txt")
test_label = np.loadtxt("test_label_real.txt")
test_data = np.loadtxt("test_data_real.txt")

total_class = 36
iterations = 3000
train_size = len(train_data)
test_size = len(test_data)
train_data = np.reshape(train_data, [train_size, 20, 30])
print(test_size)
print(train_size)
train_data = np.expand_dims(train_data, 3)
test_data = np.reshape(test_data, [test_size, 20, 30])

test_data = np.expand_dims(test_data, 3)
# print(train_data)

# #-------------------------------Preprocessing-------------------------------------#


charToClass = {48:0, 49:1, 50:2, 51:3, 52:4, 53:5, 54:6, 55:7, 56:8, 57:9, 65:10, 66:11, 67:12, 
               68:13, 69:14, 70:15, 71:16, 72:17, 73:18, 74:19, 75:20, 76:21, 77:22, 78:23, 79:24, 80:25,
               81:26, 82:27, 83:28, 84:29, 85:30, 86:31, 87:32, 88:33, 89:34, 90:35}

ClassToChar = {0:48, 1:49, 2:50, 3:51, 4:52, 5:53, 6:54, 7:55, 8:56, 9:57, 10:65, 11:66, 12:67, 
               13:68, 14:69, 15:70, 16:71, 17:72, 18:73, 19:74, 20:75, 21:76, 22:77, 23:78, 24:79, 25:80,
               26:81, 27:82, 28:83, 29:84, 30:85, 31:86, 32:87, 33:88, 34:89, 35:90}

train_data = np.float32(train_data) / 255
def toclass(label):
    return charToClass[label]
vfunc = np.vectorize(toclass)
train_label = vfunc(train_label)
test_label = vfunc(test_label)
# print(train_label)

# train_data = np.float32(train_data) / 255
# sum_x = np.sum(np.sum(np.sum(np.sum(train_data, axis=0), axis=0),axis=0),axis=0) / (train_size * 20 * 30)
# train_data = train_data - sum_x
# print(train_data)
# sum_x = np.sum(np.sum(np.sum(np.sum(train_data, axis=0), axis=0),axis=0),axis=0) / (train_size * 32 * 32 * 3)
# train_data = train_data - sum_x

# test_data = np.float32(test_data) / 255
# sum_x_test = np.sum(np.sum(np.sum(np.sum(test_data, axis=0), axis=0),axis=0),axis=0) / (test_size * 32 * 32 * 3)
# test_data = test_data - sum_x_test

# #-------------------------------Initializing-------------------------------------#
# print("Initialization...")

# # convert label to one-hot format, here the tf.
train_y_oneHot = np.zeros([train_size,36])
for i in range(train_size):
    train_y_oneHot[i,train_label[i]] = 1

test_y_oneHot = np.zeros([test_size, 36])
for i in range(test_size):
    test_y_oneHot[i,test_label[i]] = 1

# train_batch = np.reshape(train_data, [total_batch,batch_size,32,32,3])
# train_label_batch = np.reshape(train_label, [total_batch,batch_size])
# train_y_oneHot_batch = np.reshape(train_y_oneHot, [total_batch,batch_size,10])

sess = tf.Session()
X = tf.placeholder("float", [None,20,30,1])
Y = tf.placeholder("float",[None])
Y_array = tf.placeholder("float", [None,36])

X_te = tf.placeholder("float", [None, 20, 30, 1])
Y_te = tf.placeholder("float", [None])
Y_te_array = tf.placeholder("float", [None, 36])

w0_dim = [5,5,1,32]
bias0_dim = [32]

w1_dim = [3,3,32,64]
bias1_dim = [64]

# w2_dim = [3,3,32,64]
# bias2_dim = [64]
w3_dim = [960, 36] # 576 is the dimension of P_arrow
bias3_dim = [36]

dimensions = []
dimensions.append(w0_dim)
dimensions.append(bias0_dim)
dimensions.append(w1_dim)
dimensions.append(bias1_dim)

dimensions.append(w3_dim)
dimensions.append(bias3_dim)

W_pools = [1,2,2,1]
strd = [1,1,1,1]

W = []
# initialize weights for convolutional layers
for i in range(4):
    oui = tf.get_variable(name = "W" + str(i), shape = dimensions[i], dtype = "float", 
                    initializer = tf.contrib.layers.xavier_initializer())
    W.append(oui)
# initialize weights for output layers
wout = tf.get_variable(name = "Wout", shape = dimensions[4], dtype="float", 
                       initializer = tf.constant_initializer(0.1))
bias_out = tf.get_variable(name = "bias_out", shape = dimensions[5], dtype="float",
                       initializer = tf.constant_initializer(0.1))
W.append(wout)
W.append(bias_out)
itah = tf.placeholder("float", shape=())
sess.run(tf.global_variables_initializer())

# # write a function to train
#     # input : W, W_pools, strd, train_size, test_size, total_class, X, Y, Y_array, X_te, Y_te, Y_te_array, iterations
#     # output : W, cost_tr, err_tr, cost_te, error_te, predic_op(test)
# #-------------------------------Forward Propagation-------------------------------------#
[cost, err] = forward(W,W_pools,strd,train_size,total_class,X,Y,Y_array)
[cost_e, err_e] = forward(W,W_pools,strd,test_size,total_class,X_te,Y_te,Y_te_array)
predict_op = Test(W,W_pools,strd,1,total_class,X_te)

# #-----------------------------------Training-----------------------------------------#
# print("Start training...")
optimizer = tf.train.GradientDescentOptimizer(itah).minimize(cost)
# batch_index = np.random.randint(total_batch, size = iterations)

training_loss = []
training_error = []
test_loss = []
test_error = []
cur_time = time.time()
increase_threshold = 1000;
mini_error = 1.0

steps = []
for i in range(iterations):
    x = train_data
    y = train_label
    y_array = train_y_oneHot
    rate = 5 / np.sqrt((i + 1))
    # if rate < 0.05:
    #     rate = 0.05
    sess.run(optimizer, feed_dict = {X:x, Y:y, Y_array:y_array, itah:rate})
    # print(i)
    # print("%05d-th iteration, training loss = %3.6f, training error = %0.4f" % ((i+1),loss_tr,err_tr))
    if (i+1) % 10 == 0:
        [loss_tr,err_tr] = sess.run([cost,err], feed_dict = {X:x, Y:y, Y_array:y_array})
        [loss_te,err_te] = sess.run([cost_e,err_e], feed_dict = {X_te:test_data, Y_te:test_label, Y_te_array: test_y_oneHot})
        print("%05d-th iteration, training loss = %3.6f, training error = %0.4f, test error = %0.4f" % ((i+1),loss_tr,err_tr, err_te))
        
        training_error.append(err_tr)
        training_loss.append(loss_tr)
        test_error.append(err_te)
        # test_loss.append(loss_te)
        steps.append(i / 100)

        # early stop if test error is obviously increasing
        # if err_te < mini_error :
        #     mini_error = err_te
        # if (err_te - mini_error) > 0.2:
        #     break

# #-----------------------------------Saving-----------------------------------------#
# # save the test_data placeholder and prediction operation
# # pass a matrix of dimension [20, 30]

print("Saving model")
tf.get_collection("validation_nodes")
tf.add_to_collection("validation_nodes", X_te)
tf.add_to_collection("validation_nodes", predict_op)
saver = tf.train.Saver()
save_path = saver.save(sess, "my_model")

# #-----------------------------------Plotting-----------------------------------------#
print("Training costs %f minutes" % ((time.time() - cur_time) / 60))

plt.xlabel("Iterations")
plt.ylabel("Error Rate")
plt.plot(steps,training_error,"r-",label = "Training Error")
plt.plot(steps,test_error, "k-",label = "Test Error")
plt.show()
