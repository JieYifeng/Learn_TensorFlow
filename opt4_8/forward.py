import tensorflow as tf
def get_weight(shape,regularizer):
	w = tf.Variable(tf.random_normal(shape),dtype=tf.float32)
	tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
	return w

def get_bias(shape):
	b = tf.Variable(tf.constant(0.01,shape=shape))
	return b

def forward(x,regularizer):
	wl= get_weight([2,11],regularizer)
	bl= get_bias([11])
	yl= tf.nn.relu(tf.matmul(x,wl)+bl)

	w2=get_weight([11,1],regularizer)
	b2=get_bias([1])
	y=tf.matmul(yl,w2)+b2
	return y