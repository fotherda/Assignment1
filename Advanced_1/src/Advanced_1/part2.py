
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import Advanced_1.confusionMatrix as cm

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.core.protobuf import saver_pb2
from Advanced_1.dataBatcher import DataBatcher
from Advanced_1.network import Network
from Advanced_1.layer import LinearLayer
from Advanced_1.layer import ReLULayer
from Advanced_1.layer import ConvLayer
from Advanced_1.layer import ConvLayerColumns
from Advanced_1.layer import MaxPoolLayer
from Advanced_1.layer import FlattenLayer

from Advanced_1.layer import CrossEntropyLogits
from Advanced_1.convergenceTester import ConvergenceTester


from scipy.misc import toimage
from sklearn.metrics import confusion_matrix


root_dir = 'C:/Users/Dave/Documents/GI13-Advanced/Assignment1';
summaries_dir = root_dir + '/Summaries';
save_dir = root_dir + '/SavedVariables';

def print_confusion_matrix(network, X, y, model_type):
    
    y_pred = network.get_predictions(X)
    y_true = y.argmax(axis=1)
    
    cnf_matrix = confusion_matrix(y_true, y_pred)
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    cm.plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title=('Confusion matrix - ' + model_type))    
    plt.show()


def part_a(network):
    network.addLayer( LinearLayer(10,(10,784),10) )
    return 'P2_a'
   
def part_b(network):
    network.addLayer( ReLULayer(128,(128,784),128) )
    network.addLayer( LinearLayer(10,(10,128),10) )
    return 'P2_b'
   
def part_c(network):
    network.addLayer( ReLULayer(256,(256,784),256) )
    network.addLayer( ReLULayer(256,(256,256),256) )
    network.addLayer( LinearLayer(10,(10,256),10) )
    return 'P2_c'
   
def part_d(network, X_train, X_test, use_saved=True):
    
    cl1 = ConvLayerColumns(depth=16, filter_size=3, width=28, height=28)
    
    if use_saved:
        X_train_cols = np.load('X_train_cols.npy')
        X_test_cols = np.load('X_test_cols.npy')
    else:
        X_train_pad = cl1.pre_pad_all_images(X_train)
        X_train_cols = cl1.im2col(X_train_pad)
        np.save('X_train_cols', X_train_cols)
        
        X_test_pad = cl1.pre_pad_all_images(X_test)
        X_test_cols = cl1.im2col(X_test_pad)
        np.save('X_test_cols', X_test_cols)
           
    network.addLayer( cl1 )
    network.addLayer( MaxPoolLayer(2, 14, 14, (4,14,14,16), 16 )) 
    network.addLayer( ConvLayer(depth=16, filter_size=3, width=14, height=14) ) 
    network.addLayer( MaxPoolLayer(2, 7, 7, (4,7,7,16), 16) ) 
    network.addLayer( FlattenLayer((16,7,7)) ) 
    network.addLayer( ReLULayer(256, (256,784), 256) ) 
    network.addLayer( LinearLayer(10,(10,256),10) )

    return 'P2_d', X_train_cols, X_test_cols   

   
def main(_):    
        
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    X_train = mnist.train.images
    y_train = mnist.train.labels
    X_test = mnist.test.images
    y_test = mnist.test.labels   

    network = Network()
    network.addLossFunction( CrossEntropyLogits() )
    

    if False:
        model_type, X_train_cols, X_test_cols = part_d(network, X_train, X_test, use_saved=True)
        data_batcher = DataBatcher(X_train_cols, y_train)
    else:
#         model_type = part_a(network)
#         model_type = part_b(network)
        model_type = part_c(network)
        data_batcher = mnist.train
        
#     rs = np.reshape(mnist.test.images[0], (28,28))
#     toimage(rs).show()

    train_accuracy_hist = []
    test_accuracy_hist = []
    conv_tester = ConvergenceTester(0.0005, lookback_window=5) #stop if converged to within 0.05%
    learning_rate = 0.0001
    
    for i in range(30000):
        batch_xs, batch_ys = data_batcher.next_batch(50)
        train_error = network.run_one_train_epoch(batch_xs, batch_ys, learning_rate)

        if i % 100 == 0:
            train_accuracy = network.get_accuracy(X_train, y_train)
            test_accuracy = network.get_accuracy(X_test, y_test)
            train_accuracy_hist.append([i,train_accuracy])
            test_accuracy_hist.append([i,test_accuracy])

            print('{0:d} accuracy train test: {1:0.5f} : {2:0.5f}'.format( i, train_accuracy, test_accuracy))
            
            if conv_tester.has_converged(test_accuracy):
                print('converged after ', i, ' epochs')
                break
        
    np.savetxt(model_type + "_train_accuracy_hist.csv", train_accuracy_hist, delimiter=",", fmt='%f')
    np.savetxt(model_type + "_test_accuracy_hist.csv", test_accuracy_hist, delimiter=",", fmt='%f')

    print_confusion_matrix(network, X_train, y_train, model_type + ' train data')
    print_confusion_matrix(network, X_test, y_test, model_type + ' test data')
    
    

    
#     cross_entropy = tf.reduce_mean(
#         tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
#     tf.summary.scalar('CrossEntropy', cross_entropy)
#     
#     learningRate = 0.05;
#     train_step = tf.train.GradientDescentOptimizer(learningRate).minimize(cross_entropy)
# 
#     # Test trained model
#     argm_y = tf.argmax(y, 1)
#     argm_y_ = tf.argmax(y_, 1)
#     correct_prediction = tf.equal(argm_y, argm_y_)
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     tf.summary.scalar('accuracy', accuracy)
# 
#     X_train = mnist.train.images
#     y_train = mnist.train.labels
#     X_test = mnist.test.images
#     y_test = mnist.test.labels   
#       
#              
#     if False:        
#         with tf.Session() as sess:
#             file_name = save_dir + '/lRate_' + str(learningRate) + '.ckpt';    
#             saver2restore = tf.train.Saver()
#             tf.global_variables_initializer().run()    
#             saver2restore.restore(sess, file_name)
#             
#             print("Restored values:")
#             print_confusion_matrix(x, y_, X_test, y_test, argm_y, argm_y_, 
#                                    sess, 'Test', keep_prob)
#             
# 
#     with tf.Session() as sess:
# #         sess = tf.InteractiveSession()
#     
#         # Merge all the summaries and write them out to file
#         merged = tf.summary.merge_all()
#         
#         rootDir = summaries_dir + '/lRate_' + str(learningRate);
#         train_writer = tf.summary.FileWriter(rootDir + '/train', sess.graph)
#         test_writer = tf.summary.FileWriter(rootDir + '/test')
#         
#         # init operation
#         tf.global_variables_initializer().run()    
#         
#         
#         # Train
#         for i in range(20000):
#             batch_xs, batch_ys = mnist.train.next_batch(50)
#             summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
#             train_writer.add_summary(summary, i)
#      
#             if i % 100 == 0:
#                 train_accuracy = accuracy.eval(feed_dict={
#                     x:batch_xs, y_: batch_ys, keep_prob: 1.0})
#                 print("step %d, training accuracy %g" % (i, train_accuracy))
#                                   
#                 # Test trained model
#                 summary = sess.run(merged, feed_dict={x: X_test,
#                                                     y_: y_test, keep_prob: 1.0})
#                 test_writer.add_summary(summary, i)
#      
#         
#         print_confusion_matrix(x, y_, X_train, y_train, argm_y, argm_y_, 
#                                sess, 'Train', keep_prob)
#         print_confusion_matrix(x, y_, X_test, y_test, argm_y, argm_y_, 
#                                sess, 'Test', keep_prob)
#                 
#         print("Training Error rate:", 1-accuracy.eval({x: X_train, y_: y_train, 
#                                                               keep_prob: 1.0}))
#         print("Test Error rate:", 1-accuracy.eval({x: X_test, y_: y_test, 
#                                                               keep_prob: 1.0}))
#         
# #         saver = tf.train.Saver(write_version=saver_pb2.SaverDef.V2)
# #         saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
#         saver = tf.train.Saver()
#         file_name = save_dir + '/lRate_' + str(learningRate) + '.ckpt';
#         save_path = saver.save(sess, file_name)
#         print("Model saved in file: %s" % save_path)
#     
     
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

