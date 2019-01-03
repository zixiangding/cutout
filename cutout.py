import numpy as np
import tensorflow as tf

def cutout(img, num_holes=1, length=4):
  """
  Args:
  img (Tensor): Tensor image of size (H, W, C). input is an image
  Returns:
  Tensor: Image with n_holes of dimension length x length cut out of it.
  """

  h = img.shape[0]
  w = img.shape[1]
  c = img.shape[2]
  mask = np.ones([h, w], np.float32)
  for _ in range(num_holes):
    y = np.random.randint(h)
    x = np.random.randint(w)
    y1 = np.clip(max(0, y - length // 2), 0, h)
    y2 = np.clip(max(0, y + length // 2), 0, h)
    x1 = np.clip(max(0, x - length // 2), 0, w)
    x2 = np.clip(max(0, x + length // 2), 0, w)
    mask[y1: y2, x1: x2] = 0
  mask = tf.expand_dims(mask, 2)
  mask_list = []
  for _ in range(c):
    mask_list.append(mask)
  mask = tf.concat(mask_list, axis=2)
  img = img * mask
  return img


input  = tf.get_variable('v1', [1,10,10,3], dtype=tf.float32)
print(input)
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(input.get_shape()[0]):
    output = cutout(input[i], num_holes=1, length=4)
    print(output.eval())


