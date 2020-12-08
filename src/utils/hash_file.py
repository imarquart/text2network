import hashlib
import os

def check_step(folder,hash):
    if os.path.exists(''.join([folder,'/',hash,'.txt'])):
        return True
    else:
        return False

def complete_step(folder, hash):
    if not os.path.exists(folder): os.mkdir(folder)
    x=''.join([folder,'/',hash,'.txt'])
    open(x, 'a').close()
    return

def hash_file(filename, hash_factory="md5", chunk_num_blocks=128):
    """
    Calculates a hash
    :param file: file
    :param hash_factory: "md5" or "blake2"
    :return: hash: The hashed string
    """

    if hash_factory=="blake2":
        hash_factory=hashlib.blake2b
    else:
        hash_factory=hashlib.md5

    h = hash_factory()
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_num_blocks * h.block_size), b''):
            h.update(chunk)
    return h.hexdigest()

def hash_string(text, hash_factory="md5"):
    """
    Calculates a hash
    :param file: file
    :param hash_factory: "md5" or "blake2"
    :return: hash: The hashed string
    """

    if hash_factory=="blake2":
        hash_factory=hashlib.blake2b
    else:
        hash_factory=hashlib.md5

    h = hash_factory()
    h.update(text.encode('utf-8'))
    return h.hexdigest()