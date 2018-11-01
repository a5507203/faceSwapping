import tensorflow as tf
import numpy as np

IMAGE_SHAPE = (None,64, 64, 3)
ENCODER_DIM = 1024

encoderH5 = 'encoder.h5'
decoder_AH5 = 'decoder_A.h5'
decoder_BH5 = 'decoder_B.h5'


class Model(object):

    def __init__(self, model_dir = './save_model'):

        self.initModel()
        # self.autoencoder_B = None
        # self.autoencoder_A = None
        # self.saver = None
        self.checkpoint_path = ''

    def load_model(self):
        checkpoint = tf.train.get_checkpoint_state(self.checkpoint_path)
        if checkpoint:


            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print('.............Model restored to global.............')
        else:
            print('................No model is found.................')

    def save_model(self, time_step):
        print('............save model ............')
        self.saver.save(self.sess, self.checkpoint_path + '/'+ str(time_step) + '.ckpt')


    def initModel(self):


        with tf.device("/cpu:0"):

            optimizer = tf.train.AdamOptimizer(learning_rate=5e-5, beta1=0.5, beta2=0.999)

            self.obs = tf.placeholder(tf.float32,shape=IMAGE_SHAPE)
            self.target = tf.placeholder(tf.float32,shape=IMAGE_SHAPE)
            self.encoder = self.Encoder(self.obs)
            self.decoder_A = self.Decoder(self.encoder)
            self.decoder_B = self.Decoder(self.encoder)


            self.loss_A = tf.reduce_mean(tf.abs(self.obs - self.decoder_A))
            self.loss_B = tf.reduce_mean(tf.abs(self.obs - self.decoder_B))

            self.train_op_A = optimizer.minimize(loss=self.loss_A, global_step=tf.train.get_global_step())
            self.train_op_B = optimizer.minimize(loss=self.loss_B, global_step=tf.train.get_global_step())


            self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver(max_to_keep=5)
        self.sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter("./logs/", self.sess.graph)

    def converter(self, swap = False):
        autoencoder = self.decoder_B if not swap else self.decoder_A
        return lambda img: self.sess.run(autoencoder,feed_dict={self.obs:img})

    def conv(self, filters, x):
        return tf.layers.conv2d( x, filters, kernel_size=5, strides=2, padding='same',activation=my_leaky_relu)

    def upscale(self, filters, x):
        x = tf.layers.conv2d(x, filters * 4, kernel_size=3, padding='same', activation=my_leaky_relu)
        x = self.PixelShuffler(x)
        return x

    def Encoder(self, x):
        x = self.conv(128, x)
        x = self.conv(256, x)
        x = self.conv(512, x)
        x = self.conv(1024, x)
        x = tf.layers.dense(tf.layers.flatten(x), ENCODER_DIM)
        print(x)
        x = tf.layers.dense(x,4 * 4 * 1024)
        print(x)
        x = tf.reshape(x,(-1,4, 4, 1024))
        print(x)

        x = self.upscale(512, x)
        return x

    def Decoder(self, x):
        x = self.upscale(256, x)
        x = self.upscale(128, x)
        x = self.upscale(64, x)
        x = tf.layers.conv2d(x, 3, kernel_size=5, padding='same', activation=tf.nn.sigmoid)
        return x


    def train_on_batch(self, warped_A, target_A, warped_B, target_B):
        self.sess.run(self.train_op_A,feed_dict={self.obs:warped_A,self.target:target_A})
        print('asdfsadfsd')
        self.sess.run(self.train_op_B, feed_dict={self.obs: warped_B, self.target: target_B})

        loss_A = self.sess.run(self.loss_A, feed_dict={self.obs: warped_A, self.target: target_A})
        loss_B = self.sess.run(self.loss_B, feed_dict={self.obs: warped_B, self.target: target_B})

        return loss_A,loss_B

    def PixelShuffler(self,inputs):

        input_shape = inputs.get_shape().as_list()
        size = np.array((2,2))

        batch_size, h, w, c = input_shape
        if batch_size is None:
            batch_size = -1
        rh, rw = size
        oh, ow = h * rh, w * rw
        oc = c // (rh * rw)

        out = tf.reshape(inputs, (batch_size, h, w, rh, rw, oc))
        out = tf.transpose(out, (0, 1, 3, 2, 4, 5))
        out = tf.reshape(out, (batch_size, oh, ow, oc))

        return out

def my_leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=0.2)