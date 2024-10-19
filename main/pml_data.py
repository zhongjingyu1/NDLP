import numpy as np
import torch

'''
    :args: batch_size,dataset,data_dir,partial_rate,imb_type,imb_ratio,seed,hierarchical,data_dir_prod
'''
def generate_uniform_cv_candidate_labels(train_labels, partial_rate=0.1):
    train_labels = torch.from_numpy(train_labels).squeeze(1)
    n = train_labels.shape[0]
    K = train_labels.shape[1]

    partialY = train_labels
    p_1 = partial_rate
    transition_matrix = train_labels.numpy().copy()
    transition_matrix[transition_matrix!=1]=p_1

    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        partialY[j, :] = torch.from_numpy((random_n[j, :] < transition_matrix[j, :]) * 1)

    partialY = partialY.unsqueeze(1).numpy()

    print("Finish Generating Candidate Label Sets!\n")
    return partialY