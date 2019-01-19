import tensorflow as tf
from tensorflow.contrib.eager.python import tfe


def gru(units):
    return tf.keras.layers.GRU(units,
                               return_sequences=True,
                               return_state=True,
                               recurrent_activation='sigmoid',
                               recurrent_initializer='glorot_uniform')


class DNN_Encoder(tf.keras.Model):
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(DNN_Encoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.fc1 = tf.keras.layers.Dense(embedding_dim *2)
        self.fc2 = tf.keras.layers.Dense(embedding_dim)
        self.checkpoint = tfe.Checkpoint(optimizer=tf.train.AdamOptimizer(),
                                         encoder=self,
                                         optimizer_step=tf.train.get_or_create_global_step())


    def call(self, x):
        x = self.fc1(x)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        x = tf.nn.relu(x)
        return x

    def save_model(self, checkpoint_path):
        """ saves the model to a file """
        self.checkpoint.save(file_prefix=checkpoint_path)

    def load_model(self, checkpoint_path):
        """ loads the model from a file """
        self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))


class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.units = units
        self.vocab_size = vocab_size

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = gru(self.units)
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.checkpoint = tfe.Checkpoint(optimizer=tf.train.AdamOptimizer(),
                                         decoder=self,
                                         optimizer_step=tf.train.get_or_create_global_step())


    def call(self, x, features, hidden):
        x = self.embedding(x)
        x = tf.concat([tf.cast(tf.expand_dims(features, 1), tf.float32), x], axis=-1)
        output, state = self.gru(x)
        x = self.fc1(output)
        x = tf.reshape(x, (-1, x.shape[2]))
        x = self.fc2(x)

        return x, state

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


    def save_model(self, checkpoint_path):
        """ saves the model to a file """
        self.checkpoint.save(file_prefix=checkpoint_path)


    def load_model(self, checkpoint_path):
        """ loads the model from a file """
        self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))

