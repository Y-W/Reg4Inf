import tensorflow as tf

def n1(std):
    std = float(std)
    def noise_fn_n1(x):
        return tf.multiply(x, tf.random_normal(x.shape, mean=1.0, stddev=std, name='noise'))
    return noise_fn_n1

def n2(prob):
    keep_prob = 1.0 - float(prob)
    def noise_fn_n2(x):
        return tf.nn.dropout(x, keep_prob, name='noise')
    return noise_fn_n2

def identity(x):
    return x

def make_noise_fn(noise_type, noise_param):
    if noise_type is None:
        return identity
    elif noise_type == 'n1':
        return n1(noise_param)
    elif noise_type == 'n2':
        return n2(noise_param)
    else:
        raise NotImplementedError('Noise type not recognized')
