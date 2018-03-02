import numpy as np
import pandas as pd
import tensorflow as tf
import datetime

from util.get_files import get_file
from convolutional_neural_network import config


READ_DATA = False
SAVE_DATA = True


class CNN:
    def __init__(self):
        self.__data = None  # type: pd.DataFrame
        self.__activity = None  # type: np.ndarray
        self.__person = None  # type: np.ndarray
        self.__X = None
        self.__Y = None
        self.__loss = None
        self.__optimizer = None
        self.__accuracy = None
        self.__train_x = None
        self.__train_y = None
        self.__test_x = None
        self.__test_y = None

    def read_data(self):
        self.__activity = None
        for f, p, a in get_file():
            ndf = pd.read_csv(f, header=None).values.reshape(1, 1, 125, 45)  # type: np.ndarray
            self.__data = np.concatenate((self.__data, ndf), axis=0) \
                if self.__data is not None \
                else ndf

            ndf = np.array([a])
            self.__activity = np.vstack((self.__activity, ndf)) \
                if self.__activity is not None \
                else ndf

            ndf = np.array([p])
            self.__person = np.vstack((self.__person, ndf)) \
                if self.__person is not None \
                else ndf

    def save_data(self, path):
        np.save(path + 'data.npy', self.__data)
        np.save(path + 'person.npy', self.__person)
        np.save(path + 'activity.npy', self.__activity)

    def load_data(self, path):
        self.__data = np.load(path + 'data.npy')
        self.__person = np.load(path + 'person.npy')
        self.__activity = np.load(path + 'activity.npy')

    def split_data(self):
        dummies = np.asarray(pd.get_dummies(pd.DataFrame(self.__activity)))

        train_test_split = np.random.rand(len(self.__data)) < 0.70
        self.__train_x = self.__data[train_test_split]
        self.__train_y = dummies[train_test_split]
        self.__test_x = self.__data[~train_test_split]
        self.__test_y = dummies[~train_test_split]

    def build_model(self):
        def __weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def __bias_variable(shape):
            initial = tf.constant(0.0, shape=shape)
            return tf.Variable(initial)

        def __depthwise_conv2d(x, w):
            return tf.nn.depthwise_conv2d(x, w, [1, 1, 1, 1], padding='VALID')

        def __apply_depthwise_conv(x, kernel_size, num_channels, depth):
            weights = __weight_variable([1, kernel_size, num_channels, depth])
            biases = __bias_variable([depth * num_channels])
            return tf.nn.relu(tf.add(__depthwise_conv2d(x, weights), biases))

        def __apply_max_pool(x, kernel_size, stride_size):
            return tf.nn.max_pool(x,
                                  ksize=[1, 1, kernel_size, 1],
                                  strides=[1, 1, stride_size, 1],
                                  padding='VALID')

        config.TOTAL_BATCHES = self.__train_x.shape[0] // config.BATCH_SIZE

        self.__X = tf.placeholder(tf.float32,
                                  shape=[None,
                                         config.HEIGHT,
                                         config.WIDTH,
                                         config.CHANNEL])
        self.__Y = tf.placeholder(tf.float32,
                                  shape=[None, config.ACTIVITIES])
        c = __apply_depthwise_conv(self.__X,
                                   config.KERNEL_SIZE,
                                   config.CHANNEL,
                                   config.DEPTH)
        p = __apply_max_pool(c, 20, 2)
        c = __apply_depthwise_conv(p,
                                   6,
                                   config.DEPTH * config.CHANNEL,
                                   config.DEPTH // 10)

        shape = c.get_shape().as_list()
        c_flat = tf.reshape(c, [-1, shape[1] * shape[2] * shape[3]])

        f_weights_l1 = __weight_variable([shape[1] *
                                          shape[2] *
                                          config.DEPTH *
                                          config.CHANNEL *
                                          (config.DEPTH // 10),
                                          config.NUM_HIDDEN])
        f_biases_l1 = __bias_variable([config.NUM_HIDDEN])
        f = tf.nn.tanh(tf.add(tf.matmul(c_flat,
                                        f_weights_l1),
                              f_biases_l1))

        out_weights = __weight_variable([config.NUM_HIDDEN,
                                         config.ACTIVITIES])
        out_biases = __bias_variable([config.ACTIVITIES])
        y_ = tf.nn.softmax(tf.matmul(f, out_weights) + out_biases)

        self.__loss = -tf.reduce_sum(self.__Y * tf.log(y_))
        self.__optimizer = tf\
            .train\
            .GradientDescentOptimizer(learning_rate=config.LEARNING_RATE)\
            .minimize(self.__loss)

        correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(self.__Y, 1))
        self.__accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def run(self):
        with tf.Session() as session:
            cost_history = np.empty(shape=[1], dtype=float)
            tf.global_variables_initializer().run()
            print('{}: '.format(datetime.datetime.now()))
            for epoch in range(config.TRAINING_EPOCHS):
                for b in range(config.TOTAL_BATCHES):
                    offset = (b * config.BATCH_SIZE) %\
                             (self.__train_y.shape[0] - config.BATCH_SIZE)
                    batch_x = self.__train_x[offset:(offset + config.BATCH_SIZE), :, :, :]
                    batch_y = self.__train_y[offset:(offset + config.BATCH_SIZE), :]
                    _, c = session.run([self.__optimizer,
                                        self.__loss],
                                       feed_dict={self.__X: batch_x,
                                                  self.__Y: batch_y})
                    cost_history = np.append(cost_history, c)
                    # print("Epoch:",
                    #       epoch + 1,
                    #       " Training Loss:",
                    #       c,
                    #       " Training Accuracy:",
                    #       session.run(self.__accuracy,
                    #                   feed_dict={self.__X: self.__train_x,
                    #                              self.__Y: self.__train_y}),
                    #       " Testing Accuracy:",
                    #       session.run(self.__accuracy,
                    #                   feed_dict={self.__X: self.__test_x,
                    #                              self.__Y: self.__test_x})
                    #       )
                print('{}. {}: '.format((epoch + 1), datetime.datetime.now()), end='')
                print("Training Accuracy:",
                      session.run(self.__accuracy,
                                  feed_dict={self.__X: self.__train_x,
                                             self.__Y: self.__train_y}), end=' ')
                print("Testing Accuracy:",
                      session.run(self.__accuracy,
                                  feed_dict={self.__X: self.__test_x,
                                             self.__Y: self.__test_y}))


def cnn_model_fn(features, labels, mode):
    # Layer 1: Input
    input_layer = tf.reshape(features, [-1, 28, 28, 1])

    # Layer 2: Convolutional 1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu
    )
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2, 2],
        strides=2
    )

    # Layer 3: Convolutional 2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu
    )
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2, 2],
        strides=2
    )

    # Layer 4: Dense layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(
        inputs=pool2_flat,
        units=1024,
        activation=tf.nn.relu
    )
    dropout = tf.layers.dropout(
        inputs=dense,
        rate=0.4,
        training=mode == tf.estimator.ModeKeys.TRAIN
    )

    # Layer 5: Logits Layer
    logits = tf.layers.dense(
        inputs=dropout,
        units=10
    )

    predictions = {
        "classes": tf.nn.softmax(logits=logits),
        "probabilities": tf.nn.softmax(logits=logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,
                                           logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op
        )

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops=eval_metric_ops
    )


def new_cnn_main():
    har = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = har.train.images
    train_labels = np.asarray(har.train.labels, dtype=np.int32)
    eval_data = har.test.images
    eval_labels = np.asarray(har.test.labels, dtype=np.int32)

    har_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir='/tmp/har_convnet_model'
    )

    tensors_to_log = {"probabilites": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50
    )

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True
    )
    har_classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook]
    )

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False
    )
    eval_results = har_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


def old_main():
    cnn = CNN()
    if READ_DATA:
        cnn.read_data()
        print('read')
        if SAVE_DATA:
            cnn.save_data('./convolutional_neural_network/')
            print('save')
    else:
        cnn.load_data('./convolutional_neural_network/')
        print('load')
    cnn.split_data()
    cnn.build_model()
    cnn.run()

    print(1)

