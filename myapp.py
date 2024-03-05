import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import io
from sklearn.decomposition import PCA, TruncatedSVD

def perform_svd(image_url, top_k):
    image = io.imread(image_url)
    U, sing_vals, V = np.linalg.svd(image)
    sigma = np.zeros(shape=image.shape)
    np.fill_diagonal(sigma, sing_vals)

    if top_k > 0:
        trunc_U = U[:, :top_k]
        trunc_sigma = sigma[:top_k, :top_k]
        trunc_V = V[:top_k, :]
        reconstructed_image = trunc_U @ trunc_sigma @ trunc_V
    else:
        reconstructed_image = np.zeros_like(image)

    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Исходное')
    ax[1].imshow(reconstructed_image, cmap='gray')
    ax[1].set_title('После SVD')

    st.pyplot(fig)

k_value = st.slider('Значение K', 0, 300, 100)

def main():
    st.title('Singular Value Decomposition')
    st.markdown('Загрузите изображение и выполните SVD.')
    image_url = st.text_input('Введите URL изображения')

    if image_url:
        perform_svd(image_url, k_value)

if __name__ == '__main__':
    main()




