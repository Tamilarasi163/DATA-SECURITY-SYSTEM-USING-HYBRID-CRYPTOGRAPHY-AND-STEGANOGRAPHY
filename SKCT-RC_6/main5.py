import cv2
import numpy as np
import random
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.linalg import hadamard

def key_schedule(key):
    w = 32
    r = 20
    P = 0xB7E15163
    Q = 0x9E3779B9

    key_words = [int.from_bytes(key[i:i+4], byteorder='big') for i in range(0, len(key), 4)]
    S = [(P + (i * Q)) & 0xFFFFFFFF for i in range(2 * r + 4)]

    A = B = i = j = 0

    v = 3 * max(len(S), len(key_words))

    for _ in range(v):
        A = S[i] = rol((S[i] + A + B) & 0xFFFFFFFF, 3)
        B = key_words[j] = rol((key_words[j] + A + B) & 0xFFFFFFFF, (A + B) & 0x1F)
        i = (i + 1) % len(S)
        j = (j + 1) % len(key_words)

    return S

def rol(x, y):
    y = y % 32  # Ensure shift count is within valid range
    return ((x << y) | (x >> (32 - y))) & 0xFFFFFFFF

def ror(x, y):
    y = y % 32  # Ensure shift count is within valid range
    return ((x >> y) | (x << (32 - y))) & 0xFFFFFFFF

def encrypt_block(block, round_keys, r):
    A = int.from_bytes(block[:4], byteorder='big')
    B = int.from_bytes(block[4:], byteorder='big')

    A = (A + round_keys[0]) & 0xFFFFFFFF
    B = (B + round_keys[1]) & 0xFFFFFFFF

    for i in range(1, r + 1):
        A = (rol((A ^ B), B) + round_keys[2*i]) & 0xFFFFFFFF
        B = (rol((B ^ A), A) + round_keys[2*i + 1]) & 0xFFFFFFFF

    A = (A + round_keys[2*r + 2]) & 0xFFFFFFFF
    B = (B + round_keys[2*r + 3]) & 0xFFFFFFFF

    encrypted_block = A.to_bytes(4, byteorder='big') + B.to_bytes(4, byteorder='big')
    return encrypted_block

def decrypt_block(block, round_keys, r):
    A = int.from_bytes(block[:4], byteorder='big')
    B = int.from_bytes(block[4:], byteorder='big')

    B = (B - round_keys[2*r + 3]) & 0xFFFFFFFF
    A = (A - round_keys[2*r + 2]) & 0xFFFFFFFF

    for i in range(r, 0, -1):
        B = ror((B - round_keys[2*i + 1]) & 0xFFFFFFFF, A) ^ A
        A = ror((A - round_keys[2*i]) & 0xFFFFFFFF, B) ^ B

    B = (B - round_keys[1]) & 0xFFFFFFFF
    A = (A - round_keys[0]) & 0xFFFFFFFF

    decrypted_block = A.to_bytes(4, byteorder='big') + B.to_bytes(4, byteorder='big')
    return decrypted_block

def pad_data(data, block_size):
    padding_len = block_size - len(data) % block_size
    padded_data = data + bytes([padding_len] * padding_len)
    return padded_data

def unpad_data(data):
    padding_len = data[-1]
    unpadded_data = data[:-padding_len]
    return unpadded_data

def encrypt_data(key, data):
    block_size = 8
    w = 16
    r = 20

    round_keys = key_schedule(key)
    encrypted_blocks = []
    padded_data = pad_data(data, block_size)

    for i in range(0, len(padded_data), block_size):
        block = padded_data[i:i+block_size]
        encrypted_block = encrypt_block(block, round_keys, r)
        encrypted_blocks.append(encrypted_block)

    encrypted_data = b''.join(encrypted_blocks)
    return encrypted_data

def decrypt_data(key, encrypted_data):
    block_size = 8
    w = 16
    r = 20

    round_keys = key_schedule(key)
    decrypted_blocks = []

    for i in range(0, len(encrypted_data), block_size):
        block = encrypted_data[i:i+block_size]
        decrypted_block = decrypt_block(block, round_keys, r)
        decrypted_blocks.append(decrypted_block)

    decrypted_data = b''.join(decrypted_blocks)
    unpadded_data = unpad_data(decrypted_data)
    return unpadded_data

import tkinter as tk
from tkinter import filedialog 
# Encryption function
def encrypt():
      
    # img1 and img2 are the
    # two input images
    img1 = cv2.imread('pic1.jpg')
    img2 = cv2.imread('pic2.jpg')
        
    # Load an image (you can replace 'image.jpg' with your image file)
    image = plt.imread('pic2.jpg')

    # Ensure the image is grayscale
    if len(image.shape) == 3:
        image = image.mean(axis=2)

    # Perform the Hadamard Transform
    hadamard_matrix_size = image.shape[0]  # Assumes the image is square
    hadamard_matrix = hadamard(hadamard_matrix_size)
    transformed_image = np.dot(hadamard_matrix, np.dot(image, hadamard_matrix.T))

    # Inverse Hadamard Transform (for reconstruction)
    reconstructed_image = np.dot(hadamard_matrix.T, np.dot(transformed_image, hadamard_matrix))
    # Normalize pixel values to [0, 255]
    reconstructed_image = (reconstructed_image - np.min(reconstructed_image)) / (np.max(reconstructed_image) - np.min(reconstructed_image)) * 255
    # Convert to uint8 data type (required for cv2.imwrite)
    reconstructed_image = reconstructed_image.astype(np.uint8)
    cv2.imwrite("dht.jpg",reconstructed_image)

    # Display the original and transformed images
    plt.figure(figsize=(10, 5))
    plt.subplot(131)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')

    plt.subplot(132)
    plt.title('Transformed Image')
    plt.imshow(transformed_image, cmap='gray')

    plt.subplot(133)
    plt.title('Reconstructed Image')
    plt.imshow(reconstructed_image, cmap='gray')

    plt.show()
    img2 = cv2.imread('dht.jpg')
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            for l in range(3):
                  
                # v1 and v2 are 8-bit pixel values
                # of img1 and img2 respectively
                v1 = format(img1[i][j][l], '08b')
                v2 = format(img2[i][j][l], '08b')
                  
                # Taking 4 MSBs of each image
                v3 = v1[:4] + v2[:4] 
                  
                img1[i][j][l]= int(v3, 2)
                  
    cv2.imwrite('pic3in2.png', img1)
    with open('pic3in2.png', 'rb') as f:
        image_data = f.read()
    key = b'0000000000000000'
    


    encrypted_data = encrypt_data(key, image_data)
    print("Encrypted Data Length:", len(encrypted_data))
    print("Encrypted Data (Hex):", encrypted_data.hex())
    with open('encrypted_image.enc', 'wb') as f:
        f.write(encrypted_data)
  
      
      
# Driver's code
encrypt()

