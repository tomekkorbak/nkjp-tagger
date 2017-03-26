import numpy as np
import tensorflow as tf


def xavier_weight_init():
    """Returns function that creates random tensor.

    The specified function will take in a shape (tuple or 1-d array) and must
    return a random tensor of the specified shape and must be drawn from the
    Xavier initialization distribution.

    """

    def _xavier_initializer(shape, **kwargs):
        """Defines an initializer for the Xavier distribution.

        This function will be used as a variable scope initializer.

        Args:
          shape: Tuple or 1-d array that species dimensions of requested tensor.
        Returns:
          out: tf.Tensor of specified shape sampled from Xavier distribution.
        """
        epsilon = np.sqrt(6) / np.sqrt(sum(shape))
        out = tf.random_uniform(shape, minval=-epsilon, maxval=epsilon,
                                seed=1337)
        return out

    return _xavier_initializer


def data_iterator(orig_X, orig_y=None, batch_size=32, tagset_size=2):
    data_X = orig_X
    data_y = orig_y
    total_processed_examples = 0
    total_steps = int(np.ceil(len(data_X) / float(batch_size)))
    for step in xrange(total_steps):
        # Create the batch by selecting up to batch_size elements
        batch_start = step * batch_size
        x = data_X[batch_start:batch_start + batch_size]
        # Convert our target from the class index to a one hot vector
        y = None
        if np.any(data_y):
            y_indices = data_y[batch_start:batch_start + batch_size]
            y = np.zeros((len(x), tagset_size), dtype=np.int32)
            y[np.arange(len(y_indices)), y_indices] = 1
        ###
        yield x, y
        total_processed_examples += len(x)
    # Sanity check to make sure we iterated over all the dataset as intended
    assert total_processed_examples == len(
        data_X), 'Expected {} and processed {}'.format(len(data_X),
                                                       total_processed_examples)


def invert_dict(d):
    return {v: k for k, v in d.iteritems()}
