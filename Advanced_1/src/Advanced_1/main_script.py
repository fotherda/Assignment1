from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tensorflow as tf

import Advanced_1.part1 as p1
import Advanced_1.part2 as p2
import Advanced_1.confusionMatrix as cm

def main(_): 
    
#     print(os.path.dirname(inspect.getfile(tensorflow)))
#     cm.compare_confusion_matrices()
    
    if FLAGS.model[:2]=='P1':
        p1.run_part1_models(FLAGS)
    elif FLAGS.model[:2]=='P2':
        p2.run_part2_models(FLAGS)  
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    parser.add_argument('-saved_model_dir', type=str, default='C:/Users/Dave/Documents/GI13-Advanced/Assignment1/SavedModels',
                        help='Directory where trained models are saved')
    parser.add_argument('-use_saved', action='store_true', help='Use saved data')
    parser.add_argument('--model', type=str, default='P1_a', 
        help='which model to run, one of [P1_a, P1_b, P1_c, P1_d, P2_a, P2_b, P3_c, P4_d]')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
