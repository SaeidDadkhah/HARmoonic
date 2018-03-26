import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
import pickle
import os

from util.files import get_file


class CNN:
    def __init__(self, config_address=None):
        # config attributes
        self.CHECKPOINT_STEP = 1

        self.ACTIVITIES = 19

        self.HEIGHT = 1
        self.WIDTH = 125
        self.CHANNEL = 45

        self.BATCH_SIZE = 10
        self.KERNEL_SIZE = 10
        self.DEPTH = 15
        self.NUM_HIDDEN = 1000

        self.LEARNING_RATE = 0.0001
        self.TRAINING_EPOCHS = 3

        self.TOTAL_BATCHES = 1

        self.training_accuracy = list()
        self.testing_accuracy = list()
        self.cost_history = np.empty(shape=[1], dtype=float)

        # load config
        try:
            self.__load_config(config_address.format('config') + '.pkl')
        except FileNotFoundError:
            pass
        except AttributeError:
            pass

        # attributes
        self.__data = None  # type: pd.DataFrame
        self.__activity = None  # type: np.ndarray
        self.__person = None  # type: np.ndarray
        self.__X = None
        self.__Y = None
        self.__result = None
        self.__loss = None
        self.__optimizer = None
        self.__accuracy = None
        self.__train_x = None
        self.__train_y = None
        self.__test_x = None
        self.__test_y = None

    def read_data(self, root_dir=os.sep.join(['.', 'data', ''])):
        self.__activity = None
        for f, p, a in get_file(root_dir):
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

    def __save_config(self, path):
        with open(path, 'wb') as output_file:
            config = {
                "TRAINING_ACCURACY": self.training_accuracy,
                "TESTING_ACCURACY": self.testing_accuracy,
                "COST_HISTORY": self.cost_history,
                "CHECKPOINT_STEP": self.CHECKPOINT_STEP,
                "ACTIVITIES": self.ACTIVITIES,
                "HEIGHT": self.HEIGHT,
                "WIDTH": self.WIDTH,
                "CHANNEL": self.CHANNEL,
                "BATCH_SIZE": self.BATCH_SIZE,
                "KERNEL_SIZE": self.KERNEL_SIZE,
                "DEPTH": self.DEPTH,
                "NUM_HIDDEN": self.NUM_HIDDEN,
                "LEARNING_RATE": self.LEARNING_RATE,
                "TRAINING_EPOCHS": self.TRAINING_EPOCHS,
                "TOTAL_BATCHES": self.TOTAL_BATCHES,
            }
            pickle.dump(config, output_file)

    def __load_config(self, path):
        with open(path, 'rb') as input_file:
            config = pickle.load(input_file)
            self.training_accuracy = config["TRAINING_ACCURACY"]
            self.testing_accuracy = config["TESTING_ACCURACY"]
            self.cost_history = config["COST_HISTORY"]
            self.CHECKPOINT_STEP = config["CHECKPOINT_STEP"]
            self.ACTIVITIES = config["ACTIVITIES"]
            self.HEIGHT = config["HEIGHT"]
            self.WIDTH = config["WIDTH"]
            self.CHANNEL = config["CHANNEL"]
            self.BATCH_SIZE = config["BATCH_SIZE"]
            self.KERNEL_SIZE = config["KERNEL_SIZE"]
            self.DEPTH = config["DEPTH"]
            self.NUM_HIDDEN = config["NUM_HIDDEN"]
            self.LEARNING_RATE = config["LEARNING_RATE"]
            self.TRAINING_EPOCHS = config["TRAINING_EPOCHS"]
            self.TOTAL_BATCHES = config["TOTAL_BATCHES"]

    def save_data(self, path=os.sep.join(['.', 'convolutional_neural_network', ''])):
        np.save(path + 'data.npy', self.__data)
        np.save(path + 'person.npy', self.__person)
        np.save(path + 'activity.npy', self.__activity)

    def load_data(self, path=os.sep.join(['.', 'convolutional_neural_network', ''])):
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

    def __build_model(self):
        # noinspection PyShadowingNames
        def __weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        # noinspection PyShadowingNames
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

        try:
            self.TOTAL_BATCHES = self.__train_x.shape[0] // self.BATCH_SIZE
        except AttributeError:
            pass

        tf.reset_default_graph()

        self.__X = tf.placeholder(tf.float32,
                                  shape=[None,
                                         self.HEIGHT,
                                         self.WIDTH,
                                         self.CHANNEL])
        self.__Y = tf.placeholder(tf.float32,
                                  shape=[None, self.ACTIVITIES])
        c = __apply_depthwise_conv(self.__X,
                                   self.KERNEL_SIZE,
                                   self.CHANNEL,
                                   self.DEPTH)
        p = __apply_max_pool(c, 20, 2)
        c = __apply_depthwise_conv(p,
                                   6,
                                   self.DEPTH * self.CHANNEL,
                                   self.DEPTH // 10)

        shape = c.get_shape().as_list()
        c_flat = tf.reshape(c, [-1, shape[1] * shape[2] * shape[3]])

        f_weights_l1 = __weight_variable([shape[1] *
                                          shape[2] *
                                          self.DEPTH *
                                          self.CHANNEL *
                                          (self.DEPTH // 10),
                                          self.NUM_HIDDEN])
        f_biases_l1 = __bias_variable([self.NUM_HIDDEN])
        f = tf.nn.tanh(tf.add(tf.matmul(c_flat,
                                        f_weights_l1),
                              f_biases_l1))

        out_weights = __weight_variable([self.NUM_HIDDEN,
                                         self.ACTIVITIES])
        out_biases = __bias_variable([self.ACTIVITIES])
        y_ = tf.nn.softmax(tf.matmul(f, out_weights) + out_biases)
        self.__result = tf.add(tf.argmax(y_, axis=1), 1, name='result')
        self.__loss = -tf.reduce_sum(self.__Y * tf.log(y_))
        self.__optimizer = tf \
            .train \
            .GradientDescentOptimizer(learning_rate=self.LEARNING_RATE) \
            .minimize(self.__loss)

        correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(self.__Y, 1))
        self.__accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def run(self,
            path=os.sep.join(['.', 'convolutional_neural_network', 'model_{}.ckpt']),
            last_checkpoint=None):
        config_address = path.format('config') + '.pkl'
        try:
            self.__load_config(config_address)
        except FileNotFoundError:
            del self.training_accuracy[:]
            del self.testing_accuracy[:]
            self.__save_config(config_address)

        self.__build_model()

        with tf.Session() as session:
            if last_checkpoint is None:
                tf.global_variables_initializer().run()
                starting_epoch = 0
            elif last_checkpoint == 'final':
                return
            else:
                saver = tf.train.Saver()
                saver.restore(session, path.format(last_checkpoint))
                starting_epoch = int(last_checkpoint)
                print('Continue from {} epoch...'.format(starting_epoch))
            print('{}: '.format(datetime.datetime.now()))
            for epoch in range(starting_epoch, self.TRAINING_EPOCHS):
                for b in range(self.TOTAL_BATCHES):
                    offset = (b * self.BATCH_SIZE) % \
                             (self.__train_y.shape[0] - self.BATCH_SIZE)
                    batch_x = self.__train_x[offset:(offset + self.BATCH_SIZE), :, :, :]
                    batch_y = self.__train_y[offset:(offset + self.BATCH_SIZE), :]
                    _, c = session.run([self.__optimizer,
                                        self.__loss],
                                       feed_dict={self.__X: batch_x,
                                                  self.__Y: batch_y})
                    self.cost_history = np.append(self.cost_history, c)
                print('{}. {}: '.format((epoch + 1), datetime.datetime.now()), end='')
                self.training_accuracy.append(session.run(self.__accuracy,
                                                          feed_dict={self.__X: self.__train_x,
                                                                     self.__Y: self.__train_y}))
                print("Training Accuracy:", self.training_accuracy[-1], end=' ')
                self.testing_accuracy.append(session.run(self.__accuracy,
                                                         feed_dict={self.__X: self.__test_x,
                                                                    self.__Y: self.__test_y}))
                print("Testing Accuracy:", self.testing_accuracy[-1])
                if (epoch + 1) % self.CHECKPOINT_STEP == 0 and epoch + 1 != self.TRAINING_EPOCHS:
                    saver = tf.train.Saver()
                    saver.save(session, path.format(epoch + 1))
                    self.__save_config(config_address)
            saver = tf.train.Saver()
            saver.save(session, path.format('final'))
            self.__save_config(config_address)

    def predict(self,
                model=os.sep.join(['.', 'convolutional_neural_network', 'model_{}.ckpt']),
                path=os.sep.join(['.', 'data', 'a01', 'p1', 's01.txt']),
                last_checkpoint='final'):
        print(model, path)

        config_address = model.format('config') + '.pkl'
        self.__load_config(config_address)

        self.__build_model()

        ndf = pd.read_csv(path, header=None).values.reshape(1, 1, 125, 45)  # type: np.ndarray
        with tf.Session() as session:
            saver = tf.train.Saver()
            saver.restore(session, model.format(last_checkpoint))
            return session.run(self.__result, {self.__X: ndf})


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

    tensors_to_log = {"probabilities": "softmax_tensor"}
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
    read_data = False
    save_data = True
    train = True
    predict = True
    cnn = CNN()
    if train:
        if read_data:
            cnn.read_data()
            print('read')
            if save_data:
                cnn.save_data('./convolutional_neural_network/')
                print('save')
        else:
            cnn.load_data('./convolutional_neural_network/')
            print('load')
        cnn.split_data()
        cnn.run(path='./cnn_test/model_{}.ckpt')
    if predict:
        print(cnn.predict())
