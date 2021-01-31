import numpy as np
from scipy.misc import imsave
from scipy.misc import imresize

# this function is borrowed from https://github.com/hwalsuklee/tensorflow-mnist-AAE/blob/master/plot_utils.py
class plot_samples():
    def __init__(self, DIR, n_img_x=8, n_img_y=8, img_w=28, img_h=28):
        self.DIR = DIR
        assert n_img_x > 0 and n_img_y > 0
        self.n_img_x = n_img_x
        self.n_img_y = n_img_y
        self.n_total_imgs = n_img_x * n_img_y
        assert img_w > 0 and img_h > 0
        self.img_w = img_w
        self.img_h = img_h

    def save_images(self, images, name='result.jpg'):
        images = images.reshape(self.n_img_x*self.n_img_y, self.img_h, self.img_w)
        imsave(self.DIR + "/"+name, self._merge(images, [self.n_img_y, self.n_img_x]))

    def _merge(self, images, size):
        h, w = images.shape[1], images.shape[2]

        img = np.zeros((h * size[0], w * size[1]))

        for idx, image in enumerate(images):
            i = int(idx % size[1])
            j = int(idx / size[1])
            image_ = imresize(image, size=(w,h), interp='bicubic')
            img[j*h:j*h+h, i*w:i*w+w] = image_
        return img
