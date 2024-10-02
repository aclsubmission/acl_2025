import matplotlib.pyplot as plt
import numpy as np
import tqdm
import random
import os
import pickle
import time
from multiprocessing import Pool, cpu_count

def process_rank(args):
    RANK, X_vecs, Y_vecs, X_test_vecs, Y_test_vecs, FULL_B = args
    start_time = time.time()
    
    num_features = len(X_vecs[0])
    num_responses = len(Y_vecs[0])
    num_samples = len(X_vecs)

    U_prev = np.eye(num_features, RANK)
    V_prev = np.eye(RANK, num_responses)
    op_norm = 100

    Z_ = np.matmul(X_vecs, U_prev)
    LAMBDA = 0.1
    MDN = np.matmul(Y_vecs.transpose(), Z_)
    U_, S, V_ = np.linalg.svd(MDN, full_matrices=False)

    V_prev = np.matmul(U_, V_)

    cross_prod = X_vecs.T @ X_vecs
    regularized_matrix = cross_prod + LAMBDA * np.eye(num_features)
    chol_decomp = np.linalg.cholesky(regularized_matrix)
    inv_chol = np.linalg.inv(chol_decomp)
    inv_matrix = inv_chol.T @ inv_chol
    pre_V = inv_matrix @ X_vecs.T @ Y_vecs

    p = 0
    
    while op_norm > 1e-6 and p < 5000:
        p += 1
        U_curr = U_prev
        for ii in range(0, RANK):
            U_curr[:, ii] = pre_V @ V_prev[:, ii]        
        Z_curr = np.matmul(X_vecs, U_curr)
        MDN = np.matmul(Y_vecs.transpose(), Z_curr)
        U_, S_, V_ = np.linalg.svd(MDN, full_matrices=False)
        V_curr = np.matmul(U_, V_)
        op_norm = np.linalg.norm(U_curr @ V_curr.T - U_prev @ V_prev.T, ord=2)
        V_prev = V_curr
        U_prev = U_curr
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(RANK, p, execution_time)
    
    B_reconstructed = U_prev @ V_prev.T
    frobenius_norm = np.linalg.norm(FULL_B - B_reconstructed, ord='fro')
    train_MSE = frobenius_norm / (B_reconstructed.shape[0] * B_reconstructed.shape[1])
    
    Y_pred = X_test_vecs @ U_prev @ V_prev.T
    frobenius_norm = np.linalg.norm(Y_test_vecs - Y_pred, ord='fro')
    test_MSE = frobenius_norm / (Y_test_vecs.shape[0] * Y_test_vecs.shape[1])
    
    return RANK, train_MSE, test_MSE

ROOT = 'liwc_style/'
files = os.listdir(ROOT)
output = {}

for file in files:
    print(file)
    if not file.startswith('.'):
        with open(ROOT + file, 'rb') as handle:
            liwc_style = pickle.load(handle)
        
        style_vec = []
        
        y_map = {}
        c = 0
        for item in tqdm.tqdm(liwc_style):
            style_vec.append(item[2])
            label = item[1]
            if label not in y_map:
                y_map[label] = c
                c += 1

        Y = []
        X = []
        Y_test = []
        X_test = []
        for i in tqdm.tqdm(range(0, len(style_vec))):
            item = style_vec[i]
            if i < len(style_vec) * 0.8:
                x = np.array(item)
                X.append(x)
                y = np.zeros(len(y_map))
                label = liwc_style[i][1]
                y[y_map[label]] = 1
                Y.append(np.array(y))
            else:
                x = np.array(item)
                X_test.append(x)
                y = np.zeros(len(y_map))
                label = liwc_style[i][1]
                y[y_map[label]] = 1
                Y_test.append(np.array(y))

        Y_vecs = np.array(Y)
        X_vecs = np.array(X)
        
        Y_test_vecs = np.array(Y_test)
        X_test_vecs = np.array(X_test)

        X_vecs = np.insert(X_vecs, 0, 1, axis=1)
        X_test_vecs = np.insert(X_test_vecs, 0, 1, axis=1)

        FULL_B = np.linalg.inv(X_vecs.T @ X_vecs) @ X_vecs.T @ Y_vecs

        # Prepare arguments for parallel processing
        args_list = [(RANK, X_vecs, Y_vecs, X_test_vecs, Y_test_vecs, FULL_B) 
                     for RANK in range(2, len(X_vecs[0]))]

        # Use multiprocessing to parallelize the loop
        with Pool(processes=cpu_count()) as pool:
            results = pool.map(process_rank, args_list)

        # Process results
        train_MSE = {}
        test_MSE = {}
        for RANK, train_mse, test_mse in results:
            train_MSE[RANK] = train_mse
            test_MSE[RANK] = test_mse

        output[file] = [train_MSE, test_MSE]
        
        plt.figure()
        
        # Plot scatter of RANK and test_MSE
        plt.scatter(list(test_MSE.keys()), list(test_MSE.values()))
        plt.xlabel('RANK')
        plt.ylabel('test_MSE')
        plt.title('Scatter Plot of RANK vs test_MSE')
        plt.savefig('images/' + file + '_rank_MSE.png')

# Save output to file
output_file = 'results/rank_regressions.pickle'
with open(output_file, 'wb') as handle:
    pickle.dump(output, handle)