import torch
import torch.nn.functional as F
import numpy as np

class FastRP(torch.nn.Module):
    def __init__(self, embedding_dim=128, window_size=4, normalization=True, 
                 group_size=3, input_matrix='trans', alpha=-0.5, weights=None,
                 projection_type='striped'): # <--- This MUST be here
        """
        FastRP Engine (Fully Aligned with Baseline Logic).
        """
        super(FastRP, self).__init__()
        self.d = embedding_dim
        self.k = window_size
        self.norm = normalization
        self.g = group_size
        self.input_type = input_matrix
        self.alpha = alpha
        self.proj_type = projection_type
        
        # Default weights from the BlogCatalog baseline
        if weights is None:
            self.weights = [1.0, 1.0, 7.81, 45.28] 
        else:
            self.weights = weights

    def _get_random_projection(self, num_nodes, target_dim, device):
        """Generates R based on the selected projection type."""
        
        if self.proj_type == 'gaussian':
            # Baseline: Standard Gaussian Matrix
            return torch.randn(num_nodes, target_dim, device=device)
            
        elif self.proj_type == 'striped':
            # Variant 1: Striped Sparse Matrix
            num_groups = target_dim // self.g
            row_idx = torch.arange(num_nodes, device=device).repeat_interleave(num_groups)
            random_offsets = torch.randint(0, self.g, (num_nodes * num_groups,), device=device)
            base_offsets = torch.arange(0, num_groups * self.g, step=self.g, device=device).repeat(num_nodes)
            col_idx = base_offsets + random_offsets
            values = (torch.randint(0, 2, (num_nodes * num_groups,), device=device).float() * 2) - 1
            
            # Scale by density to match variance expectations
            density = 1 / self.g
            scale = 1.0 / np.sqrt(density)
            values *= scale
            
            indices = torch.stack([row_idx, col_idx])
            shape = (num_nodes, target_dim)
            return torch.sparse_coo_tensor(indices, values, size=shape, device=device).coalesce().to_dense()
        
        else:
            raise ValueError(f"Unknown projection type: {self.proj_type}")

    def forward(self, adj_matrix, features=None):
        N = adj_matrix.shape[0]
        device = adj_matrix.device
        
        # --- STEP 0: PRE-CALCULATE DEGREES ---
        degrees = torch.sparse.sum(adj_matrix, dim=1).to_dense()
        degrees[degrees == 0] = 1.0 
        
        # --- STEP 1: PREPARE PROJECTION MATRIX (R) ---
        if features is None:
            N_curr = self._get_random_projection(N, self.d, device)
        else:
            # Hybrid Logic
            d_feat = self.d // 4
            d_struct = self.d - d_feat
            R_struct = self._get_random_projection(N, d_struct, device)
            
            F_dim = features.shape[1]
            P = torch.randn(F_dim, d_feat, device=device) * (1.0 / np.sqrt(F_dim))
            R_feat = torch.mm(features, P)
            
            N_curr = torch.cat([R_struct, R_feat], dim=1)

        # --- STEP 2: APPLY ALPHA SCALING (Node Popularity Penalty) ---
        if self.alpha is not None:
            scale_vec = torch.pow(degrees, self.alpha).unsqueeze(1)
            N_curr = N_curr * scale_vec

        # --- STEP 3: TRANSITION MATRIX PREP ---
        inv_degrees = 1.0 / degrees
        
        # --- STEP 4: ITERATION & AGGREGATION ---
        embedding_list = []
        
        for i in range(self.k):
            # 1. Compute A * N_curr
            N_next = torch.sparse.mm(adj_matrix, N_curr)
            
            if self.input_type == 'trans':
                N_next = N_next * inv_degrees.unsqueeze(1)
            
            # 2. Normalize and Scale Intermediate Step
            if self.norm:
                N_to_store = F.normalize(N_next, p=2, dim=1)
            else:
                N_to_store = N_next
                
            w = self.weights[i] if i < len(self.weights) else 1.0
            embedding_list.append(N_to_store * w)
            
            # Update state for next hop
            N_curr = N_next
            
        # Sum all weighted, normalized powers
        final_embeddings = sum(embedding_list)
        
        # --- STEP 5: FINAL STANDARDIZATION ---
        mean = final_embeddings.mean(dim=0)
        std = final_embeddings.std(dim=0)
        std[std == 0] = 1.0
        final_embeddings = (final_embeddings - mean) / std
            
        return final_embeddings