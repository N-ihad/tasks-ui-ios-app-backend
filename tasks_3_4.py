import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from sklearn.datasets import fetch_olivetti_faces
import scipy
from scipy import fftpack
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from mpl_toolkits.mplot3d import Axes3D
import collections
import pandas as pd
import time
from random import randint
from pandas.plotting import table
import dataframe_image as dfi
import base64
import os.path

matplotlib.use('agg')
df = fetch_olivetti_faces()

def plot_3(data, num_photo):
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, figsize = (15,6))
    ax4.imshow(data[num_photo[0]], cmap=plt.cm.gray)
    ax5.imshow(data[num_photo[1]], cmap=plt.cm.gray)
    ax6.imshow(data[num_photo[2]], cmap=plt.cm.gray)
    ax1.imshow(df.images[num_photo[0]], cmap=plt.cm.gray)
    ax2.imshow(df.images[num_photo[1]], cmap=plt.cm.gray)
    ax3.imshow(df.images[num_photo[2]], cmap=plt.cm.gray)
    plt.savefig("./3-4-task/" + "3_out.jpg")

def plot_3_hist(data, num_photo):
    _, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, figsize = (15,6))
    ax4.hist(data[num_photo[0]])
    ax5.hist(data[num_photo[1]])
    ax6.hist(data[num_photo[2]])
    ax1.imshow(df.images[num_photo[0]], cmap=plt.cm.gray)
    ax2.imshow(df.images[num_photo[1]], cmap=plt.cm.gray)
    ax3.imshow(df.images[num_photo[2]], cmap=plt.cm.gray)
    plt.savefig("./3-4-task/" + "hist_out.jpg")

def plot_3_hist_data(data, num_photo):
    _, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, figsize = (15,6))
    x = np.histogram(df.images[num_photo[0]], 32)[1]
    y = np.append(np.histogram(df.images[num_photo[0]], 32)[0], 0)
    ax4.bar(x, y, width=0.01)
    x = np.histogram(df.images[num_photo[1]], 32)[1]
    y = np.append(np.histogram(df.images[num_photo[1]], 32)[0], 0)
    ax5.bar(x, y, width=0.01)
    x = np.histogram(df.images[num_photo[2]], 32)[1]
    y = np.append(np.histogram(df.images[num_photo[2]], 32)[0], 0)
    ax6.bar(x, y, width=0.01)
    ax1.imshow(df.images[num_photo[0]], cmap=plt.cm.gray)
    ax2.imshow(df.images[num_photo[1]], cmap=plt.cm.gray)
    ax3.imshow(df.images[num_photo[2]], cmap=plt.cm.gray)
    plt.savefig("./3-4-task/" + "hist_data_out.jpg")

def plot3d(array_acc):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(np.array(array_acc)[:,1], np.array(array_acc)[:,0], np.array(array_acc)[:,2], color='red', depthshade=False)
    ax.set_xlabel('Test size')
    ax.set_ylabel('Param')
    ax.set_zlabel('Accurancy')
    fig.set_figwidth(9)
    fig.set_figheight(9)
    plt.savefig("./3-4-task/" + "3d_out.jpg")

def hist_data(data, count_col):
    histed_data = []
    for img in data:
        hist = np.histogram(img, count_col)
        histed_data.append(hist[0])
    return np.array(histed_data)

def scale_data(data, scale):
    scaled_data = []
    for img in data:
        shape = img.shape[0]
        width = int(shape * scale)
        dim = (width, width)
        scaled_data.append(cv2.resize(img, dim))
    return np.array(scaled_data)

def dft_data(data, matrix_size):
    dfted_data = []
    for img in data:
        dft = np.fft.fft2(img)
        dft = np.real(dft)
        dfted_data.append(dft[:matrix_size, :matrix_size])
    return np.array(dfted_data)

def dct_data(data, matrix_size):
    dcted_data = []
    for img in data:
        dct = scipy.fftpack.dct(img, axis=1)
        dct = scipy.fftpack.dct(dct, axis=0)
        dcted_data.append(dct[:matrix_size,:matrix_size])
    return np.array(dcted_data)

def gradient_data(data, height):
    gradiented_data = []
    for img in data:
        shape = img.shape[0]
        i = 1
        result = []
        while (i) * height + 2 * height <= shape:
            prev = np.array(img[i * height:(i) * height + height, :])
            next =  np.array(img[(i) * height + height:(i) * height + 2 * height, :])
            result.append(prev - next)
            i += 1
        result = np.array(result)
        result = result.reshape((result.shape[0] * result.shape[1], result.shape[2]))
        result = np.mean(result, axis=0)
        gradiented_data.append(result)
    return np.array(gradiented_data)

def get_best_param(method, params, test_sizes):
    best_score = 0
    params_acc = []
    for param in params:
        for size in test_sizes:
            neigh = KNeighborsClassifier(n_neighbors=1)
            X = method(df.images, param)
            y = df.target
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, stratify=y, random_state=24)
            X_train = X_train.reshape(X_train.shape[0],-1)
            neigh.fit(X_train, y_train)
            X_test = X_test.reshape(X_test.shape[0],-1)
            y_predicted = neigh.predict(X_test)
            final_acc = accuracy_score(y_predicted, y_test)
            params_acc.append([param, size, final_acc])
            if final_acc > best_score:
                best_params = [param, size]
                best_score = final_acc
    plot3d(params_acc)
    return best_params, best_score, params_acc

def get_predict(method, param, size):
    neigh = KNeighborsClassifier(n_neighbors=1)
    X = method(df.images, param)
    y = df.target
    X_train_use, X_test_not_use, y_train_use, y_test_not_use = train_test_split(X, y, test_size=0.3, stratify=y, random_state=24)
    X_train_not_use, X_test_use, y_train_not_use, y_test_use = train_test_split(X, y, test_size=size, stratify=y, random_state=3)
    X_train_use = X_train_use.reshape(X_train_use.shape[0],-1)
    neigh.fit(X_train_use, y_train_use)
    X_test_use = X_test_use.reshape(X_test_use.shape[0],-1)
    y_predicted = neigh.predict(X_test_use)
    return y_predicted, y_test_use

def par_system(methods, test_sizes, params):
    full_acc = []
    for size in test_sizes:
        array_cls_acc = []
        array_y_pred = []
        par_sistem_y_pred = []
        for i in range(len(methods)):
            y_pred, y_test = get_predict(methods[i], params[i], size)
            array_cls_acc.append(accuracy_score(y_pred, y_test))
            array_y_pred.append(y_pred)
        array_y_pred = np.array(array_y_pred)
        for j in range(array_y_pred.shape[1]):
            par_sistem_y_pred.append(collections.Counter(array_y_pred[:,j]).most_common(1)[0][0])
        array_cls_acc.append(accuracy_score(par_sistem_y_pred, y_test))
        full_acc.append(array_cls_acc)
    return np.array(full_acc)

def save_parallel_system():
    imagesEncoded = []
    if os.path.exists("./3-4-task/parallel_system_table.jpg"):
        with open("./3-4-task/" + "parallel_system_3d_out.jpg", "rb") as image_file:
            encodedString = base64.b64encode(image_file.read())
        imagesEncoded.append(str(encodedString))
        with open("./3-4-task/" + "parallel_system_2d_out.jpg", "rb") as image_file:
            encodedString = base64.b64encode(image_file.read())
        imagesEncoded.append(str(encodedString))
        with open("./3-4-task/" + "parallel_system_table.jpg", "rb") as image_file:
            encodedString = base64.b64encode(image_file.read())
        imagesEncoded.append(str(encodedString))
        return imagesEncoded

    methods = [hist_data, scale_data, dft_data, dct_data, gradient_data]
    test_sizes = [0.1*i for i in range(1, 10)]

    params = [2*i for i in range(1, 11)]
    best_params_gradient, best_score, array_acc = get_best_param(gradient_data, params, test_sizes)

    params = [0.1*i for i in range(5, 11)]
    best_params_scale, best_score, array_acc = get_best_param(scale_data, params, test_sizes)

    params = [2*i for i in range(1, 11)]
    best_params_dct, best_score, array_acc = get_best_param(dct_data, params, test_sizes)
    best_params_dft, best_score, array_acc = get_best_param(dft_data, params, test_sizes)

    params = [5*i for i in range(1, 11)]
    best_params_hist, best_score, array_acc = get_best_param(hist_data, params, test_sizes)

    params = [best_params_hist[0], best_params_scale[0], best_params_dft[0], best_params_dct[0], best_params_gradient[0]]
    full_acc = par_system(methods, test_sizes, params)

    x = np.array(test_sizes)*400
    y = [0 for i in test_sizes]
    y1 = [4 for i in test_sizes]
    y2 = [2 for i in test_sizes]
    y3 = [3 for i in test_sizes]
    y4 = [1 for i in test_sizes]
    y5 = [5 for i in test_sizes]

    z = full_acc[:,0]
    z1 = full_acc[:,1]
    z2 =  full_acc[:, 2]
    z3 = full_acc[:,3]
    z4 = full_acc[:,4]
    z5 = full_acc[:,5]

    fig = plt.figure()
    fig.set_figwidth(9)
    fig.set_figheight(9)

    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, label='Гистограмма яркости')
    ax.plot(x, y1, z1, label='Scale')
    ax.plot(x, y2, z2, label='DFT')
    ax.plot(x, y3, z3, label='DCT')
    ax.plot(x, y4, z4, label='Градиентный метод')
    ax.plot(x, y5, z5, label='Параллельная система')

    ax.set_xlabel('Количество тестовых изображений')
    ax.set_ylabel('Метод')
    ax.set_zlabel('Точность')

    ax.legend()
    plt.savefig("./3-4-task/" + "parallel_system_3d_out.jpg")

    x = np.array(test_sizes)*400

    z = full_acc[:,0]
    z1 = full_acc[:,1]
    z2 =  full_acc[:, 2]
    z3 = full_acc[:,3]
    z4 = full_acc[:,4]
    z5 = full_acc[:,5]

    fig = plt.figure()


    fig, (ax) = plt.subplots(figsize = (15,6))
    ax.plot(x, z, label='Гистограмма яркости')
    ax.plot(x, z1, label='Scale')
    ax.plot(x, z2, label='DFT')
    ax.plot(x, z3, label='DCT')
    ax.plot(x, z4, label='Градиентный метод')
    ax.plot(x, z5, label='Параллельная система')

    ax.set_xlabel('Количество тестовых изображений')
    ax.set_ylabel('Точность')

    ax.legend()
    plt.savefig("./3-4-task/" + "parallel_system_2d_out.jpg")

    table = pd.DataFrame(full_acc, columns=['Гистограмма яркости', 'Scale', 'DFT', 'DCT', 'Градиентный метод', 'Параллельная система'])
    index = [40*i for i in range(1,10)]
    table.index = index
    table.index.name = 'Количество тестовых изображений'
    # df_styled = table.style.background_gradient()
    dfi.export(table,"./3-4-task/" + "parallel_system_table.jpg")

    with open("./3-4-task/" + "parallel_system_3d_out.jpg", "rb") as image_file:
        encodedString = base64.b64encode(image_file.read())

    imagesEncoded.append(str(encodedString))

    with open("./3-4-task/" + "parallel_system_2d_out.jpg", "rb") as image_file:
        encodedString = base64.b64encode(image_file.read())

    imagesEncoded.append(str(encodedString))

    with open("./3-4-task/" + "parallel_system_table.jpg", "rb") as image_file:
        encodedString = base64.b64encode(image_file.read())

    imagesEncoded.append(str(encodedString))

    return imagesEncoded
    

def save_hist_image(photos):
    new_show_df = hist_data(df.images, 10)
    plot_3_hist(new_show_df, photos)
    with open("./3-4-task/" + "hist_out.jpg", "rb") as image_file:
        encodedString = base64.b64encode(image_file.read())

    return str(encodedString)

def save_hist_hist_image(photos):
    new_show_df = hist_data(df.images, 10)
    plot_3_hist_data(new_show_df, photos)
    with open("./3-4-task/" + "hist_data_out.jpg", "rb") as image_file:
        encodedString = base64.b64encode(image_file.read())

    return str(encodedString)

def save_dft_image(photos):
    new_show_df = dft_data(df.images, 10)
    plot_3(new_show_df, photos)
    with open("./3-4-task/" + "3_out.jpg", "rb") as image_file:
        encodedString = base64.b64encode(image_file.read())

    return str(encodedString)

def save_dft_hist_image(photos):
    new_show_df = dft_data(df.images, 10)
    plot_3_hist_data(new_show_df, photos)
    with open("./3-4-task/" + "hist_data_out.jpg", "rb") as image_file:
        encodedString = base64.b64encode(image_file.read())

    return str(encodedString)

def save_dct_image(photos):
    new_show_df = dct_data(df.images, 10)
    plot_3(new_show_df, photos)
    with open("./3-4-task/" + "3_out.jpg", "rb") as image_file:
        encodedString = base64.b64encode(image_file.read())

    return str(encodedString)

def save_dct_hist_image(photos):
    new_show_df = dct_data(df.images, 10)
    plot_3_hist_data(new_show_df, photos)
    with open("./3-4-task/" + "hist_data_out.jpg", "rb") as image_file:
        encodedString = base64.b64encode(image_file.read())

    return str(encodedString)


def save_scale_image(photos, param = 0.5):
    new_show_df = scale_data(df.images, param)
    plot_3(new_show_df, photos)
    with open("./3-4-task/" + "3_out.jpg", "rb") as image_file:
        encodedString = base64.b64encode(image_file.read())

    return str(encodedString)

def save_scale_hist_image(photos, param = 0.5):
    new_show_df = scale_data(df.images, param)
    plot_3_hist_data(new_show_df, photos)
    with open("./3-4-task/" + "hist_data_out.jpg", "rb") as image_file:
        encodedString = base64.b64encode(image_file.read())

    return str(encodedString)

def save_gradient_image(photos):
    new_show_df = gradient_data(df.images, 10)
    plot_3_hist(new_show_df, photos)
    with open("./3-4-task/" + "hist_out.jpg", "rb") as image_file:
        encodedString = base64.b64encode(image_file.read())

    return str(encodedString)

def save_gradient_hist_image(photos):
    new_show_df = gradient_data(df.images, 10)
    plot_3_hist_data(new_show_df, photos)
    with open("./3-4-task/" + "hist_data_out.jpg", "rb") as image_file:
        encodedString = base64.b64encode(image_file.read())

    return str(encodedString)





def save_hist_3d_image():
    test_sizes = [0.1*i for i in range(1, 10)]
    params = [5*i for i in range(1, 11)]
    best_params_hist, best_score, array_acc = get_best_param(hist_data, params, test_sizes)
    with open("./3-4-task/" + "3d_out.jpg", "rb") as image_file:
        encodedString = base64.b64encode(image_file.read())

    return best_params_hist, best_score, array_acc, str(encodedString)

def save_dft_3d_image():
    test_sizes = [0.1*i for i in range(1, 10)]
    params = [2*i for i in range(1, 11)]
    best_params_dft, best_score, array_acc = get_best_param(dft_data, params, test_sizes)
    with open("./3-4-task/" + "3d_out.jpg", "rb") as image_file:
        encodedString = base64.b64encode(image_file.read())

    return best_params_dft, best_score, array_acc, str(encodedString)

def save_dct_3d_image():
    test_sizes = [0.1*i for i in range(1, 10)]
    params = [2*i for i in range(1, 11)]
    best_params_dct, best_score, array_acc = get_best_param(dct_data, params, test_sizes)
    with open("./3-4-task/" + "3d_out.jpg", "rb") as image_file:
        encodedString = base64.b64encode(image_file.read())

    return best_params_dct, best_score, array_acc, str(encodedString)

def save_scale_3d_image():
    test_sizes = [0.1*i for i in range(1, 10)]
    params = [0.1*i for i in range(5, 11)]
    best_params_scale, best_score, array_acc = get_best_param(scale_data, params, test_sizes)
    with open("./3-4-task/" + "3d_out.jpg", "rb") as image_file:
        encodedString = base64.b64encode(image_file.read())

    return best_params_scale, best_score, array_acc, str(encodedString)

def save_gradient_3d_image():
    test_sizes = [0.1*i for i in range(1, 10)]
    params = [2*i for i in range(1, 11)]
    best_params_gradient, best_score, array_acc = get_best_param(gradient_data, params, test_sizes)
    with open("./3-4-task/" + "3d_out.jpg", "rb") as image_file:
        encodedString = base64.b64encode(image_file.read())

    return best_params_gradient, best_score, array_acc, str(encodedString)

photos = [randint(1, 75), randint(76, 150), randint(151, 250)]
# save_hist_image(photos)
# save_hist_hist_image(photos)
# save_dft_image(photos)
# save_dft_hist_image(photos)
# save_dct_image(photos)
# save_dct_hist_image(photos)
# save_scale_image(photos)
# save_scale_hist_image(photos)
# save_gradient_image(photos)
# save_gradient_hist_image(photos)

# save_parallel_system()