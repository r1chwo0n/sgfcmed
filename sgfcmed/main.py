import random
from sklearn.model_selection import KFold
from collections import defaultdict, Counter
import numpy as np
from readFile import load_data_from_folder, split_data
from sgfcmed_parallel import SGFCMedParallel

def assign_cluster_labels(train_data, membership_matrix):
    """
    ทำ majority vote เพื่อกำหนด label ให้แต่ละ cluster
    """
    cluster_labels = defaultdict(list) # สร้าง dict เก็บไว้ว่าแต่ละ cluster มี class lable(chromosome type) เป็นอะไร
    for item, memberships in zip(train_data, membership_matrix):
        max_idx = np.argmax(memberships)  # ใน train data membership value ของ cluster ไหน มีค่ามากที่สุด
        cluster_labels[max_idx].append(item['chromosome_type']) # เอา chromosome_type ใส่ใน cluster นั้น

    # ทำ majority vote ในแต่ละ cluster ว่ามี chromosome_type ไหน มากสุด
    cluster_to_label = {}
    for cluster_idx, labels in cluster_labels.items():
        majority = Counter(labels).most_common(1)[0][0]
        cluster_to_label[cluster_idx] = majority

    return cluster_to_label # ติด class label ให้แต่ละ cluster

def run_cv_with_sgfcmed(data, k=10):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    errors = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
        train_data = [data[i] for i in train_idx]
        test_data = [data[i] for i in test_idx]

        train_strings = [item['encoded_string'] for item in train_data]

        model = SGFCMedParallel(C=22, m=2.0, tol=0.1)
        model.fit(train_strings)

        # ทำ majority vote เพื่อ map cluster → label
        membership_matrix = model.membership()
        cluster_to_label = assign_cluster_labels(train_data, membership_matrix)

        # test set
        test_strings = [item['encoded_string'] for item in test_data]
        true_labels = [item['chromosome_type'] for item in test_data]
        predicted_clusters = model.predict(test_strings)
        predicted_labels = [cluster_to_label[c] for c in predicted_clusters]

        # error
        num_wrong = sum(p != t for p, t in zip(predicted_labels, true_labels))
        fold_error = num_wrong / len(test_data)
        errors.append(fold_error)

        print(f"Fold {fold+1} error: {fold_error:.4f}")

    print(f"\nAverage error across {k} folds: {np.mean(errors):.4f}")

if __name__ == "__main__":
    # Load training set
    all_data = load_data_from_folder("chromosome")
    
    train_val_data, blind_test_data = split_data(all_data, blind_ratio=0.5)
    run_cv_with_sgfcmed(train_val_data, k=10)

    # Train full model on all training data
    train_strings = [item['encoded_string'] for item in train_val_data]
    model = SGFCMedParallel(C=22, m=2.0, tol=0.1)
    model.fit(train_strings)

    # Assign labels to clusters using majority vote
    membership_matrix = model.membership()
    cluster_to_label = assign_cluster_labels(train_val_data, membership_matrix)

    # Predict on blind test
    blind_strings = [item['encoded_string'] for item in blind_test_data]
    true_labels = [item['chromosome_type'] for item in blind_test_data]
    predicted_clusters = model.predict(blind_strings)
    predicted_labels = [cluster_to_label[c] for c in predicted_clusters]

    # Evaluate
    num_wrong = sum(p != t for p, t in zip(predicted_labels, true_labels))
    error_rate = num_wrong / len(blind_test_data)
    print(f"\n Blind Test Error Rate: {error_rate:.4f}")