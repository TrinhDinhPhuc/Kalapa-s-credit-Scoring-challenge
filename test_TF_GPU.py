import tensorflow as tf

print(tf.test.is_built_with_cuda())



if tf.test.gpu_device_name():
    print('ddddddddddddddddddDefault GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("pppppppppPlease install GPU version of TF")


hello = tf.constant('hello tensorflow')
with tf.Session() as sesh:
    sesh.run(hello)
