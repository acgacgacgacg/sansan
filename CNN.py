import tensorflow as tf
import numpy as np
import warnings
from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec

warnings.filterwarnings("ignore")
learn = tf.contrib.learn
slim = tf.contrib.slim


def cnn(x, y):
    x = tf.reshape(x, [-1, 299, 50, 1])
    y = slim.one_hot_encoding(y, 9)
    net = slim.conv2d(x, 48, [5,5], scope='conv1')
    net = slim.max_pool2d(net, [2,2], scope='pool1')
    net = slim.conv2d(net, 96, [5,5], scope='conv2')
    net = slim.max_pool2d(net, [2,2], scope='pool2')
    net = slim.flatten(net, scope='flatten')
    net = slim.fully_connected(net, 512, scope='fully_connected3')
    logits = slim.fully_connected(net, 9,
            activation_fn=None, scope='fully_connected4')

    prob = slim.softmax(logits)
    loss = slim.losses.softmax_cross_entropy(logits, y)

    train_op = slim.optimize_loss(loss, slim.get_global_step(),
            learning_rate=0.001,
            optimizer='Adam')

    return {'class': tf.argmax(prob, 1), 'prob': prob},\
            loss, train_op

# path = '/Users/aldin/Documents/ML/Statistical Learning Theory/sansan-001/'
# train_X = np.load(path+'trainDataV.npy')[:-300,:,1].astype('float32')
# train_Y = np.load(path+'labelsV.npy')[:-300]
# train_Y = np.argmax(train_Y, 1)

# test_X = np.load(path+'trainDataV.npy')[-300:,:,1].astype('float32')
# test_Y = np.load(path+'labelsV.npy')[-300:]
# test_Y = np.argmax(test_Y, 1)

tf.logging.set_verbosity(tf.logging.INFO)
validation_metrics = {
    "accuracy" : MetricSpec(
        metric_fn=tf.contrib.metrics.streaming_accuracy,
        prediction_key="class")
}
validation_monitor = learn.monitors.ValidationMonitor(
        test_X,
        test_Y,
        metrics=validation_metrics,
        every_n_steps=100)


classifier = learn.Estimator(model_fn=cnn, model_dir='/tmp/cnn_log',
    config=learn.RunConfig(save_checkpoints_secs=10))
classifier.fit(x=train_X, y=train_Y, steps=3200, batch_size=64,
    monitors=[validation_monitor])
x = tf.placeholder(tf.float32, [None, 299*50])
y = tf.placeholder(tf.int64, [None])
cl = cnn(x,y)[0]['class']
saver = tf.train.Saver()
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('/tmp/cnn_log')
    if ckpt:
        last_model = ckpt.model_checkpoint_path
        print "load " + last_model
        saver.restore(sess, last_model)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(y, cl), tf.float32))
        test_accuracy = sess.run(accuracy, feed_dict={x: test_X, y: test_Y})
        print('Test Accuracy: %s' % test_accuracy)


