import nibabel as nib
import numpy as np
import scipy.ndimage

CLASSES_LABELS = {0: "clean", 1: "artifact"}


def normalize(img):
    m=np.mean(img)
    st=np.std(img)
    norm = (img - m) / st
    return norm


def swap_axes(img):
    img = np.swapaxes(img,0,2)
    img = img[::-1, ::-1, :]
    sh=np.asarray(img.shape)
    img = np.swapaxes(img,2,np.argmin(sh))
    return img


def pad_img(img):
    blank=np.zeros((256,256,64))
    blank[:img.shape[0],:img.shape[1],:img.shape[2]]=img
    return blank


def resize_img(img,img_shape):
    size=list(img_shape)
    zoom=[1.0*x/y for x, y in zip(size, img.shape)]
    return scipy.ndimage.zoom(img, zoom=zoom)


def load_and_reorient(path):
    img = nib.load(path)
    orig_ornt = nib.io_orientation(img.affine)
    targ_ornt = nib.orientations.axcodes2ornt("SPL")
    transform = nib.orientations.ornt_transform(orig_ornt, targ_ornt)
    img_orient = img.as_reoriented(transform)
    return img_orient.get_fdata()


def get_subj_data(p, input_size=(1,256,256,64,1)):
    img_shape=input_size[1:-1]
    img = load_and_reorient(p)
    img = swap_axes(img)
    if any(np.asarray(img.shape)>np.asarray(img_shape)):
        img=resize_img(img,img_shape)
    img = normalize(img)
    img = pad_img(img)
    subj_data = np.reshape(img,input_size)
    return subj_data.astype(np.float32, copy=False)


def collate_inferences(predictions):
    """
    collate the result of multiple inferences for final prediction
    e.g.
    predictions = [
    array([0.04948492, 0.95051503], dtype=float32),
    array([0.08604479, 0.9139552 ], dtype=float32),
    array([0.11726741, 0.8827326 ], dtype=float32),
    array([0.05771826, 0.94228166], dtype=float32)
    ]

    returns ('artifact', 100.0, 4, 4)

    :param predictions: list of numpy array of shape (1, 2)
    :return: (str, either clean or artifact, probability 0-100, count, sum)
    """
    values, counts = np.unique(np.argmax(predictions, axis=1), return_counts=True)
    i = np.argmax(counts)
    inferred_class = values[i]
    c = counts[i]
    s = np.sum(counts)
    p = 100 * c / s
    return CLASSES_LABELS[inferred_class], p, c, s