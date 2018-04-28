#coding:utf-8
"""
kobexie
2018-04-28
"""
import tensorflow as tf
import numpy as np
import os
import cv2
import time

def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
# write all example into one single tfrecords file    
def write_tfRecord(data=None, label=None, file_name='test.tfrecord'):
    """
    data: your data list or np.ndarray
    label: label list 
    file_name: the tfrecord file storage data and label
    """

    # Compress the frames using JPG and store in as a list of strings in 'frames'
    # here suppose we have 10 data with same shape [50,100,3], but label has different length, eg. OCR task.
    with tf.python_io.TFRecordWriter(file_name) as writer:
        for i in range(10):
            # prepare data and label
            data = np.ones((50, 100, 3), dtype=uint8) # np.ndarray
            label = list(np.arange(i+1)) # list of int
            
            features = {} # each example has these features
            features['label'] = _int64_feature(label)# 'label' is a feature of tf.train. BytesList
            features['data'] = _bytes_feature(data.tostring())

            # write serialized example into .tfrecords
            tfrecord_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tfrecord_example.SerializeToString())

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    # 1.read serialized examples from File Queue.
    xxxx, serialized_example = reader.read(filename_queue)

    # 2.the rules of parse tf.train.Example, FixedLenFeature/VarLenFeature/SparseFeature
    features = {}
    features['data'] = tf.FixedLenFeature([], tf.string)
    features['label'] = tf.VarLenFeature(tf.int64)

    # 3.parse examples
    parsed_features = tf.parse_single_example(serialized_example, features)

    # Decode raw string using tf.decode_raw() into  uint8 and reshape original shape
    frames = tf.decode_raw(parsed_features["data"], tf.uint8)
    frames = tf.reshape(frames, [50, 100, 3]) #[50,100,3]

    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    frames = tf.cast(frames, tf.float32) * (1. / 255) - 0.5

    # Decode a tf.VarLenFeature by tf.sparse_tensor_to_dense
    # generate (index, value, shape) triple
    label = tf.serialize_sparse(parsed_features['label']) #for tf.VarLenFeature

    return frames, label
    
def inputs(tfrecords, batch_size, num_epochs, is_sparse_label=True):
    with tf.name_scope('input'):
        # 1.push the '.tfrecords' files into File Queue.
        filename_queue = tf.train.string_input_producer([tfrecords], num_epochs=num_epochs)

        # Even when reading in multiple threads, share the filename queue.
        image, label = read_and_decode(filename_queue)

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        images_batch, labels_batch_serialized = tf.train.shuffle_batch(
                                                      [image, label], batch_size=batch_size, num_threads=2,
                                                      capacity=1000 + 3 * batch_size,
                                                      # Ensures a minimum amount of shuffling of examples.
                                                      min_after_dequeue=1000)

        # for variable length labels
        sparse_labels_batch = tf.deserialize_many_sparse(labels_batch_serialized, dtype=tf.int64)
        if is_sparse_label:
            labels_batch = sparse_labels_batch
        else:
            labels_batch = tf.sparse_tensor_to_dense(sparse_labels_batch)
    return images_batch, labels_batch

def run_training():
    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # Input images and labels.
        frames, label = inputs('test.tfrecord', batch_size=3, num_epochs=5)

        # The op for initializing the variables.
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        # Create a session for running operations in the Graph.
        sess = tf.Session()

        # Initialize the variables (the trained variables and the epoch counter).
        sess.run(init_op)

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            step = 0
            while not coord.should_stop():
                start_time = time.time()

                # Run one step of the model.  The return values are
                # the activations from the `train_op` (which is
                # discarded) and the `loss` op.  To inspect the values
                # of your ops or variables, you may include them in
                # the list passed to sess.run() and the value tensors
                # will be returned in the tuple from the call.
                #_, loss_value = sess.run([train_op, loss])
                res_frames, res_label = sess.run([frames, label])
                print res_frames.shape, res_label

                duration = time.time() - start_time

                # Print an overview fairly often.
                loss_value = 0.22
                if step % 10 == 0:
                    print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                step += 1
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (5, step))
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)
        sess.close()

if __name__=="__main__":
    write_tfRecord()
    run_training()
