import os
import numpy as np
import scipy.io as sio

def download_and_convert_flickr():
    try:
        from torch_geometric.datasets import Flickr
        from torch_geometric.utils import to_scipy_sparse_matrix
    except Exception as exc:
        raise ImportError(
            "PyTorch Geometric is required. Install with: pip install torch-geometric"
        ) from exc

    print("Downloading Flickr via PyTorch Geometric...")
    dataset = Flickr(root="/tmp/Flickr")
    data = dataset[0]

    print(f"   Nodes: {data.num_nodes}")
    print(f"   Edges: {data.num_edges}")
    print(f"   Features: {data.num_features}")

    adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)
    adj = adj + adj.T
    adj.data = np.ones_like(adj.data)

    X = data.x.numpy()
    labels = data.y.numpy()
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)

    mat_data = {
        "network": adj,
        "Attributes": X,
        "group": labels,
    }

    output_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "flickr.mat")
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sio.savemat(output_path, mat_data)
    print(f"Success! Saved to {output_path}")

if __name__ == "__main__":
    download_and_convert_flickr()
