import tensorflow as tf
import time
import numpy as np
import os

def loss_function(real, pred):
    mask = 1 - np.equal(real, 0)
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
    return tf.reduce_mean(loss_)


def train(encoder, decoder, dataset,
          BATCH_SIZE=64,
          EPOCHS=10,
          CHECKPOINT_STEP=10,
          ENCODER_CHECKPOINT_PATH = "Models/Encoder",
          DECODER_CHECKPOINT_PATH="Models/Decoder"):

    optimizer = tf.train.AdamOptimizer()

    for epoch in range(EPOCHS):
        start = time.time()
        total_loss = 0

        for (batch, (tensor_input, target)) in enumerate(dataset):
            loss = 0

            # initializing the hidden state for each batch
            # because the playlist names are not related from playlist to playlist
            hidden = decoder.reset_state(batch_size=target.shape[0])

            dec_input = tf.expand_dims([0] * BATCH_SIZE, 1)

            with tf.GradientTape() as tape:
                features = encoder(tensor_input)

                for i in range(1, target.shape[1]):
                    # passing the features through the decoder
                    predictions, hidden = decoder(dec_input, features, hidden)

                    loss += loss_function(target[:, i], predictions)

                    # using teacher forcing
                    dec_input = tf.expand_dims(target[:, i], 1)

            total_loss += (loss / int(target.shape[1]))

            variables = encoder.variables + decoder.variables

            gradients = tape.gradient(loss, variables)

            optimizer.apply_gradients(zip(gradients, variables), tf.train.get_or_create_global_step())

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             loss.numpy() / int(target.shape[1])))

        print('Epoch {} Total loss {:.4f}'.format(epoch + 1, total_loss / 40))
        print('Time taken for 1 epoch {:.2f} sec\n'.format(time.time() - start))

        if (epoch + 1) % CHECKPOINT_STEP == 0:
            encoder.save_model(ENCODER_CHECKPOINT_PATH)
            decoder.save_model(DECODER_CHECKPOINT_PATH)

            print("encoder model saved to {}".format(ENCODER_CHECKPOINT_PATH))
            print("decoder model saved to {}\n".format(DECODER_CHECKPOINT_PATH))

