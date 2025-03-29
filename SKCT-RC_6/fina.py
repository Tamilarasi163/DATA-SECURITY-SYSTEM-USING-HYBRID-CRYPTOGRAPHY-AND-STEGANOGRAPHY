import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hadamard
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash

app = Flask(__name__)
app.secret_key = 'secret'
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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
    y = y % 32
    return ((x << y) | (x >> (32 - y))) & 0xFFFFFFFF

def ror(x, y):
    y = y % 32
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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle the uploaded images
        if 'image1' not in request.files or 'image2' not in request.files:
            flash('Please upload both images.')
            return redirect(request.url)

        image1 = request.files['image1']
        image2 = request.files['image2']

        if image1.filename == '' or image2.filename == '':
            flash('No selected files.')
            return redirect(request.url)

        img1_path = os.path.join(app.config['UPLOAD_FOLDER'], image1.filename)
        img2_path = os.path.join(app.config['UPLOAD_FOLDER'], image2.filename)

        image1.save(img1_path)
        image2.save(img2_path)

        # Process the images
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        image = plt.imread(img2_path)

        if len(image.shape) == 3:
            image = image.mean(axis=2)

        hadamard_matrix_size = image.shape[0]
        hadamard_matrix = hadamard(hadamard_matrix_size)
        transformed_image = np.dot(hadamard_matrix, np.dot(image, hadamard_matrix.T))

        reconstructed_image = np.dot(hadamard_matrix.T, np.dot(transformed_image, hadamard_matrix))
        reconstructed_image = (reconstructed_image - np.min(reconstructed_image)) / (np.max(reconstructed_image) - np.min(reconstructed_image)) * 255
        reconstructed_image = reconstructed_image.astype(np.uint8)
        dht_path = os.path.join(app.config['UPLOAD_FOLDER'], "dht.jpg")
        cv2.imwrite(dht_path, reconstructed_image)

        img2 = cv2.imread(dht_path)

        for i in range(img2.shape[0]):
            for j in range(img2.shape[1]):
                for l in range(3):
                    v1 = format(img1[i][j][l], '08b')
                    v2 = format(img2[i][j][l], '08b')
                    v3 = v1[:4] + v2[:4]
                    img1[i][j][l] = int(v3, 2)

        merged_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'pic3in2.png')
        cv2.imwrite(merged_image_path, img1)

        with open(merged_image_path, 'rb') as f:
            image_data = f.read()
        key = b'0000000000000000'

        encrypted_data = encrypt_data(key, image_data)
        encrypted_path = os.path.join(app.config['UPLOAD_FOLDER'], 'encrypted_image.enc')
        with open(encrypted_path, 'wb') as f:
            f.write(encrypted_data)

        return render_template('index.html', original1=image1.filename, original2=image2.filename, transformed="dht.jpg", merged="pic3in2.png", encrypted="encrypted_image.enc")

    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/decrypt', methods=['POST'])

def decrypt():
    enc_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'encrypted_image.enc')
    with open(enc_file_path, 'rb') as f:
        encrypted_data = f.read()

    key = b'0000000000000000'
    decrypted_data = decrypt_data(key, encrypted_data)
    decrypted_path = os.path.join(app.config['UPLOAD_FOLDER'], 'decrypted_image.png')
    with open(decrypted_path, 'wb') as f:
        f.write(decrypted_data)

    return render_template('index.html', decrypted="decrypted_image.png")

if __name__ == '__main__':
    app.run(debug=True)
