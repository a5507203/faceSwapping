import cv2
import numpy 
from random import shuffle
import threading
import queue as Queue


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, prefetch=1): #See below why prefetch count is flawed
        threading.Thread.__init__(self)
        self.queue = Queue.Queue(prefetch)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        # Put until queue size is reached. Note: put blocks only if put is called while queue has already reached max size
        # => this makes 2 prefetched items! One in the queue, one waiting for insertion!
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def iterator(self):
        while True:
            next_item = self.queue.get()
            if next_item is None:
                break
            yield next_item



class TrainingDataGenerator():

    def __init__(self, coverage):
        self.random_transform_args  = {
            'rotation_range': 20,
            'zoom_range': 0.05,
            'shift_range': 0.05,
            'random_flip': 0.5,
        }
        self.coverage = coverage


    def minibatchAB(self, images, batchsize):
        batch = BackgroundGenerator(self.minibatch(images, batchsize), 1)
        for ep1, warped_img, target_img in batch.iterator():
            yield ep1, warped_img, target_img

    # A generator function that yields epoch, batchsize of warped_img and batchsize of target_img
    def minibatch(self, data, batchsize):
        length = len(data)

        assert length >= batchsize, "Number of images is lower than batch-size (Note that too few images may lead to bad training). # images: {}, batch-size: {}".format(
            length, batchsize)
        epoch = i = 0
        shuffle(data)
        while True:
            size = batchsize
            if i + size > length:
                shuffle(data)
                i = 0
                epoch += 1
            rtn = numpy.float32([self.read_image(img) for img in data[i:i + size]])
            i += size
            yield epoch, rtn[:, 0, :, :, :], rtn[:, 1, :, :, :]

    def color_adjust(self, img):
        return img / 255.0

    def read_image(self, fn):
        try:
            image = self.color_adjust(cv2.imread(fn))
        except TypeError:
            raise Exception("Error while reading image", fn)

        image = cv2.resize(image, (256, 256))
        image = self.random_transform(image, **self.random_transform_args)
        warped_img, target_img = self.random_warp(image, self.coverage)

        return warped_img, target_img

    def random_transform(self, image, rotation_range, zoom_range, shift_range, random_flip):
        h, w = image.shape[0:2]
        rotation = numpy.random.uniform(-rotation_range, rotation_range)
        scale = numpy.random.uniform(1 - zoom_range, 1 + zoom_range)
        tx = numpy.random.uniform(-shift_range, shift_range) * w
        ty = numpy.random.uniform(-shift_range, shift_range) * h
        mat = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, scale)
        mat[:, 2] += (tx, ty)
        result = cv2.warpAffine(
            image, mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
        if numpy.random.random() < random_flip:
            result = result[:, ::-1]
        return result

    # get pair of random warped images from aligned face image
    def random_warp(self, image, coverage):
        assert image.shape == (256, 256, 3)
        range_ = numpy.linspace(128 - coverage // 2, 128 + coverage // 2, 5)
        mapx = numpy.broadcast_to(range_, (5, 5))
        mapy = mapx.T

        mapx = mapx + numpy.random.normal(size=(5, 5), scale=5)
        mapy = mapy + numpy.random.normal(size=(5, 5), scale=5)

        interp_mapx = cv2.resize(mapx, (80, 80))[8:72, 8:72].astype('float32')
        interp_mapy = cv2.resize(mapy, (80, 80))[8:72, 8:72].astype('float32')

        warped_image = cv2.remap(image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)

        src_points = numpy.stack([mapx.ravel(), mapy.ravel()], axis=-1)
        dst_points = numpy.mgrid[0:65:16, 0:65:16].T.reshape(-1, 2)
        mat = umeyama(src_points, dst_points, True)[0:2]

        target_image = cv2.warpAffine(image, mat, (64, 64))

        return warped_image, target_image


def stack_images(images):
    def get_transpose_axes(n):
        if n % 2 == 0:
            y_axes = list(range(1, n - 1, 2))
            x_axes = list(range(0, n - 1, 2))
        else:
            y_axes = list(range(0, n - 1, 2))
            x_axes = list(range(1, n - 1, 2))
        return y_axes, x_axes, [n - 1]

    images_shape = numpy.array(images.shape)
    new_axes = get_transpose_axes(len(images_shape))
    new_shape = [numpy.prod(images_shape[x]) for x in new_axes]
    return numpy.transpose(
        images,
        axes=numpy.concatenate(new_axes)
    ).reshape(new_shape)






def umeyama(src, dst, estimate_scale):

    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = numpy.dot(dst_demean.T, src_demean) / num

    # Eq. (39).
    d = numpy.ones((dim,), dtype=numpy.double)
    if numpy.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = numpy.eye(dim + 1, dtype=numpy.double)

    U, S, V = numpy.linalg.svd(A)

    # Eq. (40) and (43).
    rank = numpy.linalg.matrix_rank(A)
    if rank == 0:
        return numpy.nan * T
    elif rank == dim - 1:
        if numpy.linalg.det(U) * numpy.linalg.det(V) > 0:
            T[:dim, :dim] = numpy.dot(U, V)
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = numpy.dot(U, numpy.dot(numpy.diag(d), V))
            d[dim - 1] = s
    else:
        T[:dim, :dim] = numpy.dot(U, numpy.dot(numpy.diag(d), V.T))

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * numpy.dot(S, d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * numpy.dot(T[:dim, :dim], src_mean.T)
    T[:dim, :dim] *= scale

    return T
