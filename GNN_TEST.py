#!/usr/bin/env python3
'''
7/17/2025: I want to use GNNs to predict the class of the voxel that each subhalo
belongs to. I may use this output as input to the U-net to improve predictions.
'''
from spektral.layers import GCNConv, GlobalSumPool
from spektral.data import Graph, Dataset, Loader
from spektral.utils import sp_matrix_to_sp_tensor
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
from scipy.sparse import load_npz
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
import numpy as np

class GNN_Node_Classifier(Model):
    def __init__(self, n_classes, n_features, n_hidden=64):
        super(GNN_Node_Classifier, self).__init__()
        self.conv1 = GCNConv(n_hidden, activation='relu')
        self.conv2 = GCNConv(n_hidden, activation='relu')
        self.pool = GlobalSumPool()
        self.dense = Dense(n_classes, activation='softmax')
    def call(self, inputs):
        x, a = inputs
        x = self.conv1([x, a])
        x = self.conv2([x, a])
        x = self.pool(x)
        return self.dense(x)

class SubhaloDataset(Dataset):
    def __init__(self, feature_path, adjacency_path, edge_attr_path, norm=True, **kwargs):
        self.feature_path = feature_path
        self.adjacency_path = adjacency_path
        self.edge_attr_path = edge_attr_path
        self.norm = norm
        super(SubhaloDataset, self).__init__(**kwargs)
    def read(self):
        features = np.load(self.feature_path, allow_pickle=True)[()]
        pos = features['SubhaloPos']
        vel = features['SubhaloVel']
        mass = features['SubhaloMass'].reshape(-1, 1)
        photometrics = features['SubhaloStellarPhotometrics']
        labels = features['SubhaloVoxelClass']
        x = np.concatenate([pos, vel, mass, photometrics], axis=1)
        if self.norm:
            scaler = StandardScaler()
            x = scaler.fit_transform(x)

        a = load_npz(self.adjacency_path)
        a = a.maximum(a.T)  # Ensure symmetry

        e = np.load(self.edge_attr_path, allow_pickle=True)
        return [Graph(x=x, a=a, e=e, y=labels)]

### Set up paths and parameters ###
DATA_PATH = '/ifs/groups/vogeleyGrp/data/TNG/'
FIGS_PATH = '/ifs/groups/vogeleyGrp/figs/TNG_GNN_TEST/'
feature_path = DATA_PATH + 'TNG300-subhalos1-features.npy'
adjacency_path = DATA_PATH + 'TNG300-subhalos1-adjacency_matrix.npz'
edge_attr_path = DATA_PATH + 'TNG300-subhalos1-edge_attributes.npy'
n_classes = 4 


if __name__ == "__main__":
    # Load dataset
    dataset = SubhaloDataset(feature_path, adjacency_path, edge_attr_path)
    graph = dataset[0]
    x, a, y = graph.x, graph.a, graph.y
    n_features = x.shape[1]

    # Train-test split
    idx = np.arange(x.shape[0])
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)

    # Prepare model
    model = GNN_Node_Classifier(n_classes=n_classes, n_features=n_features)
    # Compile
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # Create masks for training and testing:
    mask_train = np.zeros_like(y, dtype=bool)
    mask_test = np.zeros_like(y, dtype=bool)
    mask_train[idx_train] = True
    mask_test[idx_test] = True
    # Convert to tensors
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    a = sp_matrix_to_sp_tensor(a)
    y = tf.convert_to_tensor(y, dtype=tf.int32)
    print(f"x shape: {x.shape}, a shape: {a.shape}, y shape: {y.shape}")

    # Training loop:
    N_EPOCHS = 50
    for epoch in range(N_EPOCHS):
        with tf.GradientTape() as tape:
            preds = model([x, a], training=True)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y[mask_train], preds[mask_train])
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Evaluate on test set
        logits = model([x, a], training=False)
        preds = tf.argmax(logits, axis=1).numpy()
        acc = np.mean(preds[mask_test] == y.numpy()[mask_test])
        print(f'Epoch {epoch+1}/{N_EPOCHS}, Loss: {loss.numpy():.4f}, Test Accuracy: {acc:.4f}')

    # Save the model
    model.save('/ifs/groups/vogeleyGrp/models/gnn_subhalo_classifier_full_density.h5')
    # Make final confusion matrix
    true_labels = y.numpy()[mask_test]
    pred_labels = preds[mask_test]
    cm = confusion_matrix(true_labels, pred_labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Void', 'Wall', 'Filament', 'Halo'])
    disp.plot(cmap='viridis', values_format='d')
    disp.ax_.set_title('Confusion Matrix for GNN Subhalo Classifier')
    disp.figure_.savefig(FIGS_PATH + 'gnn_subhalo_classifier_confusion_matrix.png')
    print("Confusion matrix saved to:", FIGS_PATH + 'gnn_subhalo_classifier_confusion_matrix.png')
    print("Model training complete and saved to /ifs/groups/vogeleyGrp/models/gnn_subhalo_classifier_full_density.h5")
    print('>>> All done! <<<')