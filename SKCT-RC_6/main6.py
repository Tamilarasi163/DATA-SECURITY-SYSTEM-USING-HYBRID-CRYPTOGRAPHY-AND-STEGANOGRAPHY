import cv2
import numpy as np
import random

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

def decrypt():
      
    # Encrypted image
    key = b'0000000000000000'
    with open('encrypted_image.enc', 'rb') as f:
        encrypted_data = f.read()

    decrypted_data = decrypt_data(key, encrypted_data)
    print("Decrypted Data Length:", len(decrypted_data))
    print("Decrypted Data (Hex):", decrypted_data.hex())
    with open('decrypted_image.jpg', 'wb') as f:
        f.write(decrypted_data)
    img = cv2.imread('decrypted_image.jpg') 
    width = img.shape[0]
    height = img.shape[1]
      
    # img1 and img2 are two blank images
    img1 = np.zeros((width, height, 3), np.uint8)
    img2 = np.zeros((width, height, 3), np.uint8)
      
    for i in range(width):
        for j in range(height):
            for l in range(3):
                v1 = format(img[i][j][l], '08b')
                v2 = v1[:4] + chr(random.randint(0, 1)+48) * 4
                v3 = v1[4:] + chr(random.randint(0, 1)+48) * 4
                  
                # Appending data to img1 and img2
                img1[i][j][l]= int(v2, 2)
                img2[i][j][l]= int(v3, 2)
      
    # These are two images produced from
    # the encrypted image
    cv2.imwrite('pic2_re.png', img1)
    cv2.imwrite('pic3_re.png', img2)
      
      

decrypt()

