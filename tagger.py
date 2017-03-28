import os
import sys
import time

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from utils import xavier_weight_init, data_iterator, invert_dict
from model import Model
# from print_labels import print_labels
from preprocessing import preprocess_data


class Config(object):
    """Holds model hyperparameters and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    embed_size = 50
    batch_size = 64
    hidden_size = 200
    max_epochs = 30
    early_stopping = 2
    dropout = 0.9
    lr = 0.001
    l2 = 0.003
    window_size = 3

    def __repr__(self):
        attributes = [(a, v) for a, v in self.__class__.__dict__.items()
                      if not 'function' in str(v) \
                      and not (a.startswith('__') and a.endswith('__'))]
        representation = 'Model hyperparameters:\n'
        for key, value in attributes:
            representation += '{key}: {value}\n'.format(
                key=key, value=value
            )
        return representation



class Tagger(Model):
    """Implements a tagger model.
    """

    def load_data(self, debug=False):
        """Loads starter word-vectors and train/dev/test-split the data."""

        # Load the training set
        X, y, self.word_to_num, self.tag_to_num = preprocess_data(
            dir_path='NKJP_1.2_nltk_POS')

        self.num_to_word = invert_dict(self.word_to_num)
        self.num_to_tag = invert_dict(self.tag_to_num)
        self.tagset_size = len(self.tag_to_num)

        self.X_train, self.X_dev, self.y_train, self.y_dev = train_test_split(
            X, y, test_size=0.2)
        # A hacky way to get 3-part split from 2-part-splitting function
        self.X_dev, self.X_test, self.y_dev, self.y_test = train_test_split(
            self.X_dev, self.y_dev, test_size=0.5)

        if debug:
            self.X_train = self.X_train[:1024]
            self.y_train = self.y_train[:1024]
            self.X_dev = self.X_dev[:1024]
            self.y_dev = self.y_dev[:1024]


    def add_placeholders(self):
        """Generate placeholder variables to represent the input tensors

        These placeholders are used as inputs by the rest of the model building
        code and will be fed data during training.
        """
        self.input_placeholder = tf.placeholder(
            tf.int32, shape=[None, self.config.window_size], name='Input')
        self.labels_placeholder = tf.placeholder(
            tf.float32, shape=[None, self.tagset_size], name='Target')
        self.dropout_placeholder = tf.placeholder(tf.float32, name='Dropout')


    def create_feed_dict(self, input_batch, dropout, label_batch=None):
        """Creates the feed_dict for softmax classifier.

        Args:
          input_batch: A batch of input data.
          label_batch: A batch of label data.
        Returns:
          feed_dict: The feed dictionary mapping from placeholders to values.
        """
        feed_dict = {
            self.input_placeholder: input_batch,
        }
        if label_batch is not None:
            feed_dict[self.labels_placeholder] = label_batch
        if dropout is not None:
            feed_dict[self.dropout_placeholder] = dropout
        return feed_dict

    def add_embedding(self):
        """Add embedding layer that maps from vocabulary to vectors.

        Creates an embedding tensor (of shape
        (len(self.word_to_num), embed_size)).

        Returns:
          window: tf.Tensor of shape (-1, window_size*embed_size)
        """
        with tf.device('/cpu:0'):
            embedding = tf.get_variable(
                'Embedding',
                [len(self.word_to_num), self.config.embed_size]
            )
            window = tf.nn.embedding_lookup(embedding, self.input_placeholder)
            window = tf.reshape(
                window, [-1, self.config.window_size * self.config.embed_size])
            return window

    def add_model(self, window):
        """Adds the 1-hidden-layer NN.

         Args:
          window: tf.Tensor of shape (-1, window_size*embed_size)
        Returns:
          output: tf.Tensor of shape (batch_size, tagset_size)
        """
        with tf.variable_scope('Layer1',
                               initializer=xavier_weight_init()) as scope:
            W = tf.get_variable(
                'W', [self.config.window_size * self.config.embed_size,
                      self.config.hidden_size])
            b1 = tf.get_variable('b1', [self.config.hidden_size])
            h = tf.nn.tanh(tf.matmul(window, W) + b1)
            if self.config.l2:
                tf.add_to_collection('total_loss',
                                     0.5 * self.config.l2 * tf.nn.l2_loss(W))

        with tf.variable_scope('Layer2',
                               initializer=xavier_weight_init()) as scope:
            U = tf.get_variable('U', [self.config.hidden_size,
                                      self.tagset_size])
            b2 = tf.get_variable('b2', [self.tagset_size])
            y = tf.matmul(h, U) + b2
            if self.config.l2:
                tf.add_to_collection('total_loss',
                                     0.5 * self.config.l2 * tf.nn.l2_loss(U))
        output = tf.nn.dropout(y, self.dropout_placeholder)
        return output

    def add_loss_op(self, y):
        """Adds cross_entropy_loss ops to the computational graph.

        Args:
          pred: A tensor of shape (batch_size, n_classes)
        Returns:
          loss: A 0-d tensor (scalar)
        """
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(y, self.labels_placeholder)
        )
        tf.add_to_collection('total_loss', cross_entropy)
        loss = tf.add_n(tf.get_collection('total_loss'))
        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See

        Args:
          loss: Loss tensor, from cross_entropy_loss.
        Returns:
          train_op: The Op for training.
        """
        op = tf.train.AdamOptimizer(learning_rate=self.config.lr)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = op.minimize(loss, global_step=global_step)
        return train_op

    def __init__(self, config):
        """Constructs the network using the helper functions defined above."""
        self.config = config
        self.load_data(debug=False)
        self.add_placeholders()
        window = self.add_embedding()
        y = self.add_model(window)

        self.loss = self.add_loss_op(y)
        self.predictions = tf.nn.softmax(y)
        one_hot_prediction = tf.argmax(self.predictions, 1)
        correct_prediction = tf.equal(
            tf.argmax(self.labels_placeholder, 1), one_hot_prediction)
        self.correct_predictions = tf.reduce_sum(
            tf.cast(correct_prediction, 'int32'))
        self.train_op = self.add_training_op(self.loss)

    def run_epoch(self, session, input_data, input_labels,
                  shuffle=True, verbose=True):
        orig_X, orig_y = input_data, input_labels
        dp = self.config.dropout
        # We're interested in keeping track of the loss and accuracy
        # during training
        total_loss = []
        total_correct_examples = 0
        total_processed_examples = 0
        total_steps = len(orig_X) / self.config.batch_size
        for step, (x, y) in enumerate(
                data_iterator(orig_X, orig_y,
                              batch_size=self.config.batch_size,
                              tagset_size=self.tagset_size)):
            feed = self.create_feed_dict(input_batch=x, dropout=dp,
                                         label_batch=y)
            loss, total_correct, _ = session.run(
                [self.loss, self.correct_predictions, self.train_op],
                feed_dict=feed)
            total_processed_examples += len(x)
            total_correct_examples += total_correct
            total_loss.append(loss)
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(
                    step, total_steps, np.mean(total_loss)))
                sys.stdout.flush()
        if verbose:
            sys.stdout.write('\r')
            sys.stdout.flush()
        return np.mean(total_loss), total_correct_examples / float(
            total_processed_examples)

    def predict(self, session, X, y=None, verbose=False):
        """Make predictions from the provided model."""
        # If y is given, the loss is also calculated
        # We deactivate dropout by setting it to 1
        dp = 1
        losses = []
        results = []
        if np.any(y):
            data = data_iterator(X, y, batch_size=self.config.batch_size,
                                 tagset_size=self.tagset_size)
        else:
            data = data_iterator(X, batch_size=self.config.batch_size,
                                 tagset_size=self.tagset_size)
        for step, (x, y) in enumerate(data):
            feed = self.create_feed_dict(input_batch=x, dropout=dp)
            if np.any(y):
                feed[self.labels_placeholder] = y
                loss, preds = session.run(
                    [self.loss, self.predictions], feed_dict=feed)
                losses.append(loss)
            else:
                preds = session.run(self.predictions, feed_dict=feed)
            predicted_indices = preds.argmax(axis=1)
            results.extend(predicted_indices)
            if verbose:
                print 'SENTENCE %d:' % step
                sentences, _, labels = print_labels(x, preds, y,
                                                    self.num_to_word,
                                                    self.num_to_tag)
                print sentences
                print labels
        return np.mean(losses), results


def print_confusion(confusion, num_to_tag):
    """Helper method that prints confusion matrix."""
    # Summing top to bottom gets the total number of tags guessed as T
    total_guessed_tags = confusion.sum(axis=0)
    # Summing left to right gets the total number of true tags
    total_true_tags = confusion.sum(axis=1)
    print 'Confusion matrix (precission and recall for each tag)'
    for i, tag in sorted(num_to_tag.items()):
        prec = confusion[i, i] / float(total_guessed_tags[i])
        recall = confusion[i, i] / float(total_true_tags[i])
        print 'Tag: {} - P {:2.4f} / R {:2.4f}'.format(tag, prec, recall)


def calculate_confusion(predicted_indices, y_indices, tagset_size):
    """Helper method that calculates confusion matrix."""
    confusion = np.zeros((tagset_size, tagset_size),
                         dtype=np.int32)
    for i in xrange(len(y_indices)):
        correct_label = y_indices[i]
        guessed_label = predicted_indices[i]
        confusion[correct_label, guessed_label] += 1
    return confusion


def save_predictions(predictions, filename):
    """Saves predictions to provided file."""
    with open(filename, "wb") as f:
        for prediction in predictions:
            f.write(str(prediction) + "\n")


def test_tagger():
    """Test NER model implementation.

    You can use this function to test your implementation of the Named Entity
    Recognition network. When debugging, set max_epochs in the Config object to 1
    so you can rapidly iterate.
    """
    config = Config()
    with tf.Graph().as_default():
        model = Tagger(config)
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()

        with tf.Session() as session:
            best_val_loss = float('inf')
            best_val_epoch = 0

            session.run(init)
            for epoch in xrange(config.max_epochs):
                print 'Epoch {}'.format(epoch)
                start = time.time()
                train_loss, train_acc = model.run_epoch(session, model.X_train,
                                                        model.y_train)
                val_loss, predictions = model.predict(session, model.X_dev,
                                                      model.y_dev)
                print 'Training loss: {}'.format(train_loss)
                print 'Training acc: {}'.format(train_acc)
                print 'Dev loss: {}'.format(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_epoch = epoch
                    if not os.path.exists("./weights"):
                        os.makedirs("./weights")

                    saver.save(session, './weights/tagger.weights')
                if epoch - best_val_epoch > config.early_stopping:
                    break
                confusion = calculate_confusion(predictions,
                                                model.y_dev,
                                                tagset_size=model.tagset_size)
                print_confusion(confusion, model.num_to_tag)
                print 'Total time: {}'.format(time.time() - start)
                print 'Total time: {}'.format(time.time() - start)

                total_correct_examples = 0
                total_processed_examples = len(predictions)
                for y_hat, y in zip(predictions, model.y_dev):
                    total_correct_examples += 1 if y == y_hat else 0
                print 'Dev acc is', total_correct_examples/float(total_processed_examples)

            # saver.restore(session, './weights/tagger.weights')
            # print 'Test'
            # print '=-=-='
            # print 'Writing predictions to `preds`'
            # _, predictions = model.predict(session, model.X_test, model.y_test,
            #                                verbose=False)
            # total_correct_examples = 0
            # total_processed_examples = len(predictions)
            # for y_hat, y in zip(predictions, model.y_test):
            #     total_correct_examples += 1 if y == y_hat else 0
            # print 'Test acc is', total_correct_examples / float(
            #     total_processed_examples)

            # save_predictions(predictions, "preds")


if __name__ == "__main__":
    test_tagger()
