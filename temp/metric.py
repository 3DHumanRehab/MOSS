import numpy as np
from scipy.stats import entropy

def cosine_distance(v1, v2):
    return 1 - np.abs(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def calculate_distances(vectors_a, vectors_b):
    m, n = vectors_a.shape[0], vectors_b.shape[0]
    d_inter = np.sum([cosine_distance(v_a, v_b) for v_a in vectors_a for v_b in vectors_b]) / (m * n)
    d_intra_a = np.sum([cosine_distance(v_a_i, v_a_j) for v_a_i in vectors_a for v_a_j in vectors_a]) / (m**2)
    d_intra_b = np.sum([cosine_distance(v_b_i, v_b_j) for v_b_i in vectors_b for v_b_j in vectors_b]) / (n**2)
    d_intra = d_intra_a + d_intra_b
    return d_inter, d_intra

def calculate_relative_entropy(vectors_a, vectors_b):
    # This assumes vectors are probability distributions. Adjust as needed.
    m, n = vectors_a.shape[0], vectors_b.shape[0]
    import pdb
    pdb.set_trace()
    en_inter = np.sum([entropy(a, b) for a in vectors_a for b in vectors_b]) / (m * n)
    en_intra_a = np.sum([entropy(a_i, a_j) for a_i in vectors_a for a_j in vectors_a]) / (m**2)
    en_intra_b = np.sum([entropy(b_i, b_j) for b_i in vectors_b for b_j in vectors_b]) / (n**2)
    en_intra = en_intra_a + en_intra_b
    return en_inter, en_intra

# vectors_a = np.random.rand(10, 2)  # Replace with actual data
# vectors_b = np.random.rand(8, 2)   # Replace with actual data

path =f"Qemu_t-SNE2_vecList.txt"

# import pdb
# pdb.set_trace()
data = np.loadtxt(path)

path2 = f"label.txt"
labels= np.loadtxt(path2)
data = data[:labels.shape[0]]
 
vectors_a_mask = labels==1
vectors_a = data[vectors_a_mask]

vectors_b_mask = labels==0
vectors_b = data[vectors_b_mask]

vectors_a_min = vectors_a.min(axis=0)
vectors_a_max = vectors_a.max(axis=0)
vectors_a = (vectors_a - vectors_a_min) / (vectors_a_max - vectors_a_min)

vectors_b_min = vectors_b.min(axis=0)
vectors_b_max = vectors_b.max(axis=0)
vectors_b = (vectors_b - vectors_b_min) / (vectors_b_max - vectors_b_min)

vectors_a[vectors_a==0] = 1e-6
vectors_b[vectors_b==0] = 1e-6

# vectors_a = np.random.rand(10, 2)  # Replace with actual data
# vectors_b = np.random.rand(8, 2)   # Replace with actual data
d_inter, d_intra = calculate_distances(vectors_a, vectors_b)
print(f"D_inter: {d_inter}, D_intra: {d_intra}")
en_inter, en_intra = calculate_relative_entropy(vectors_a, vectors_b)
print(f"EN_inter: {en_inter}, EN_intra: {en_intra}")

