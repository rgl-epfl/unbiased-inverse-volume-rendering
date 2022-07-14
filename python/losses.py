import drjit as dr


def average(img, ref_img, shape=None):
    return dr.sum(img) / dr.width(img)

def l1(img, ref_img, shape=None):
    return dr.sum(dr.abs(img - ref_img)) / dr.width(img)

def l2(img, ref_img, shape=None):
    return dr.sum(dr.sqr(img - ref_img)) / dr.width(img)

def root_mean_squared_error(*args, **kwargs):
    return dr.sqrt(l2(*args, **kwargs))

def huber(img, ref_img, shape=None, delta=1.0):
    residual = img - ref_img
    loss = dr.select(residual < delta,
                     0.5 * dr.sqr(residual),
                     delta * dr.abs(residual) - 0.5 * delta)
    return dr.sum(loss) / dr.width(img)

def mean_relative_absolute_error(img, ref_img, shape=None, epsilon=1e-2):
    errors = dr.abs(img - ref_img) / (dr.abs(ref_img) + epsilon)
    return dr.sum(errors) / dr.width(img)

def mean_relative_squared_error(img, ref_img, shape=None, epsilon=1e-2):
    errors = dr.sqr(img - ref_img) / (dr.sqr(ref_img) + epsilon)
    return dr.sum(errors) / dr.width(img)

def root_mean_relative_squared_error(*args, **kwargs):
    return dr.sqrt(mean_relative_squared_error(*args, **kwargs))


def psnr(img, ref_img, max_value=1.0, shape=None):
    """Caution, this is not well-suited to HDR images."""
    # -- https://github.com/tensorflow/tensorflow/blob/c256c071bb26e1e13b4666d1b3e229e110bc914a/tensorflow/python/ops/image_ops_impl.py#L4064-L4068
    # mse = math_ops.reduce_mean(math_ops.squared_difference(a, b), [-3, -2, -1])
    # psnr_val = math_ops.subtract(
    #     20 * math_ops.log(max_val) / math_ops.log(10.0),
    #     np.float32(10 / np.log(10)) * math_ops.log(mse),
    #     name='psnr')
    # -- https://gist.github.com/nimpy/5b0085075a54ba2e94f2cfabf5a98a57
    # mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    # if mse == 0:
    #     return 100
    # return 20 * np.log10(max_value / (np.sqrt(mse)))

    mse = dr.hsum_async(dr.sqr(img - ref_img)) / dr.width(img)
    psnr = 20. * (dr.log(max_value) / dr.log(10.)) - (10. / dr.log(10.)) * dr.log(mse)
    return psnr

