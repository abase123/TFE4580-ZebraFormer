
#torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import optim
#Libaries for calculation and processing
from einops import rearrange, repeat
from math import sqrt
from math import ceil     

class FullAttention(nn.Module):
    '''
    The Attention operation
    '''
    def __init__(self, scale=None, attention_dropout=0.2,return_attention = False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)
        self.return_attention = return_attention

        
    def forward(self, queries, keys, values, mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if mask is not None:
            scores += mask.to(queries.device)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        if self.return_attention:
            return V.contiguous(), A.contiguous()
        
        return V.contiguous()
    
    
    


class AttentionLayer(nn.Module):
    '''
    The Multi-head Self-Attention (MSA) Layer
    '''
    def __init__(self, d_model, n_heads,sparsity=None, d_keys=None, d_values=None, dropout = 0.1,return_attention=False,cross_flag=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)
        
        self.return_attention = return_attention
        self.inner_attention = FullAttention(scale=None, attention_dropout = dropout,return_attention=self.return_attention)


        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.sparsity = sparsity
        self.cross_flag = cross_flag
        self.mask = None

        

    def generate_non_circular_sparse_mask(self,num_patches, num_heads):
        """
        Generate a sparse attention mask where each patch attends to itself and a specified number of its neighbors,
        which could be to the left or right, without wrapping at boundaries. The mask will not include
        circular references (no wrapping around the array edges).
        """
        # Initialize the mask with negative infinity (blocking attention)
        mask = torch.full((num_heads, num_patches, num_patches), float('-inf'))

        for h in range(num_heads):
            for i in range(num_patches):
                # Allowed patches include the current index (self-attention allowed)
                allowed_patches = [i]

                # Determine the number of neighbors to include from each side
                num_neighbors = self.sparsity // 2
                # Add neighbors to the left, checking boundaries
                for j in range(1, num_neighbors + 1):
                    if i - j >= 0:
                        allowed_patches.append(i - j)

                # Add neighbors to the right, checking boundaries
                for j in range(1, num_neighbors + 1):
                    if i + j < num_patches:
                        allowed_patches.append(i + j)

                # Set the mask to zero for allowed indices
                for j in allowed_patches:
                    mask[h, i, j] = 0

        return mask
    
    def generate_ts_d_specific_mask(num_time_series, num_segments_per_series, num_heads):
        """Generate an attention mask where each time series only attends to its own segments."""
        # Create a mask filled with negative infinity (blocks attention)
        mask = torch.full((num_heads, num_time_series * num_segments_per_series, num_time_series * num_segments_per_series), float('-inf'))
        for ts_d in range(num_time_series):
            start_index = ts_d * num_segments_per_series
            end_index = start_index + num_segments_per_series
            mask[:, start_index:end_index, start_index:end_index] = 0
        
        return mask

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
       
        
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        
        #if self.mask is None:
            #self.mask = self.generate_non_circular_sparse_mask(L, H)
       
        if(self.return_attention):
            out,attention_weights= self.inner_attention(
                queries,
                keys,
                values,
               self.mask,)

            out = out.view(B, L, -1)
        
            return self.out_projection(out), attention_weights
        
        else:
            out = self.inner_attention(
            queries,
            keys,
            values,
            self.mask,)

            out = out.view(B, L, -1)
            return self.out_projection(out)
            

        
class AttentionLayerCrossSegments(nn.Module):
    '''
    The Multi-head Self-Attention (MSA) Layer
    '''

    def __init__(self, d_model, n_heads,sparsity,d_keys=None, d_values=None,cross_flag=False, dropout = 0.1,return_attention=True):
        super(AttentionLayerCrossSegments, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)
        
        self.return_attention = return_attention
        self.inner_attention1 = FullAttention(scale=None, attention_dropout = dropout,return_attention=self.return_attention)
        self.inner_attention2 = FullAttention(scale=None, attention_dropout = dropout,return_attention=self.return_attention)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out1_projection = nn.Linear(d_values * n_heads, d_model)
        
        self.out2_projection = nn.Linear(d_values * n_heads, d_model)

        #self.out2_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.sparsity = sparsity
        self.mask = None
        self.cross_flag = cross_flag
        
        if(cross_flag):
            self.cross_attention = FullAttention(scale=None, attention_dropout = dropout,return_attention=self.return_attention)
        
    def generate_non_circular_sparse_mask(self,num_patches, num_heads):
        """
        Generate a sparse attention mask where each patch attends to itself and a specified number of its neighbors,
        which could be to the left or right, without wrapping at boundaries. The mask will not include
        circular references (no wrapping around the array edges).
        """
        # Initialize the mask with negative infinity (blocking attention)
        mask = torch.full((num_heads, num_patches, num_patches), float('-inf'))

        for h in range(num_heads):
            for i in range(num_patches):
                # Allowed patches include the current index (self-attention allowed)
                allowed_patches = [i]

                # Determine the number of neighbors to include from each side
                num_neighbors = self.sparsity // 2

                # Add neighbors to the left, checking boundaries
                for j in range(1, num_neighbors + 1):
                    if i - j >= 0:
                        allowed_patches.append(i - j)

                # Add neighbors to the right, checking boundaries
                for j in range(1, num_neighbors + 1):
                    if i + j < num_patches:
                        allowed_patches.append(i + j)

                # Set the mask to zero for allowed indices
                for j in allowed_patches:
                    mask[h, i, j] = 0

        return mask
    
    
        

    def forward(self, queries, keys, values,num_patches):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        
        queries_projected = self.query_projection(queries).view(B, L, H, -1)
        keys_projected = self.key_projection(keys).view(B, S, H, -1)
        values_projected = self.value_projection(values).view(B, S, H, -1)
    
        if(self.cross_flag):

            outC,attention_weightsC= self.cross_attention(
                queries_projected,
                keys_projected,
                values_projected,
                self.mask,
                )
            outC = outC.view(B, L, -1)
            
            return self.out2_projection(outC)

        

        self.mask = self.generate_non_circular_sparse_mask(num_patches, H)
        q = queries_projected[:, :num_patches, :, :]
        k = keys_projected[:, -num_patches:, :, :]
        v = values_projected[:, -num_patches:, :, :]
        L= q.shape[1]
        
        if(self.return_attention):
            
            out1,attention_weights1= self.inner_attention1(
                q,
                k,
                v,
                self.mask,
                )
            
            out2,attention_weights2= self.inner_attention2(
                k,
                q,
                q,
                self.mask,
                )

            out1 = out1.view(B, L, -1)
            out2 = out2.view(B, L, -1)
            concatenated = torch.cat([out1, out2], dim=1)
            
            
            return self.out1_projection(concatenated), attention_weights1, attention_weights2,
        
        
        else:
            
            out1 = self.inner_attention1(
                q,
                k,
                v,
                self.mask,
                )
            
            out2,attention_weights2= self.inner_attention2(
                k,
                q,
                q,
                self.mask,
                )
            out1 = out1.view(B, L, -1)
            out2 = out2.view(B, L, -1)
            concatenated = torch.cat([out1, out2], dim=1)
            
            return self.out1_projection(out1), self.out1_projection(out2)
        
        
        
        
class TwoStageAttentionLayerCrossSegments(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''
    def __init__(self, seg_num, factor, d_model, n_heads,sparsity,d_ff = None, dropout=0.1):
        super(TwoStageAttentionLayerCrossSegments, self).__init__()
        d_ff = d_ff or 4*d_model
        self.time_attention = AttentionLayer(d_model, n_heads,sparsity, dropout = dropout,return_attention=False)
        #self.dim_sender = AttentionLayerCrossSegments(d_model, n_heads, dropout = dropout,return_attention=True)
        #self.dim_receiver = AttentionLayer(d_model, n_heads, dropout = dropout,return_attention=True)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))
        
        self.projection_layer = nn.Linear(2 * d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm31 = nn.LayerNorm(d_model)
        self.norm32 = nn.LayerNorm(d_model)

        self.norm41 = nn.LayerNorm(d_model)
        self.norm42 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                nn.GELU(),
                                nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                nn.GELU(),
                                nn.Linear(d_ff, d_model))
        
            
        self.attention_details = []

        
    
    def store_attn(self,receive_weights,send_weights):
        
        self.attention_details.append({
            'att-A-B': send_weights.detach(),  # Detach tensors for storage
            'att-B-A': receive_weights.detach(),
        })
    
    
    def get_attn(self):
        return self.attention_details
        
    def reset_attention_across_channels_details(self):
        self.attention_details = []
        
    def forward(self, x):
        #input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
        batch = x.shape[0]
        seg_num = x.shape[2]
        ts_d = x.shape[1]
        d_model = x.shape[3]
        
        x_copy = x
        #Cross Time Stage: Directly apply MSA to each dimension
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc = self.time_attention(
            time_in, time_in, time_in
        )
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)
        
        final_out1 = rearrange(dim_in, '(b ts_d) seg_num d_model -> b ts_d seg_num d_model', b = batch,ts_d=ts_d)
     
        return  final_out1
        
        #Cross dimension segment-segment
        segments = x_copy.reshape(batch, ts_d*seg_num , d_model)
        
        dim_enc1, att1, att2 = self.dim_sender(segments,segments,segments,seg_num)
        #dim_enc1, att1 = self.dim_sender(segments,segments,segments)
        
        self.store_attn(att1,att2)
        
        dim_enc1 = segments + self.dropout(dim_enc1)
        
        dim_enc1 = self.norm31(dim_enc1)
        #dim_enc2 = self.norm32(dim_enc2)
        
        dim_enc1 = dim_enc1 + self.dropout(self.MLP2(dim_enc1))
       # dim_enc2 = dim_enc2 + self.dropout(self.MLP2(dim_enc2))
        
        dim_enc1 = self.norm41(dim_enc1)
        #dim_enc2 = self.norm42(dim_enc2)
        
       # concatenated = torch.cat([dim_enc1, dim_enc2], dim=1)
        
        final_out2 = rearrange(dim_enc1, 'b (ts_d seg_num) d_model -> b ts_d seg_num d_model', ts_d=ts_d, seg_num=seg_num)
        
        
        gate = torch.sigmoid(final_out2) 
        final_out = gate * final_out2 + (1 - gate) * final_out1

        
        return final_out2
        



class GatingLayer(nn.Module):
    def __init__(self, d_model):
        super(GatingLayer, self).__init__()
        # This feedforward network outputs two values for each attention mechanism
        self.gate = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),  # Outputs two values per input dimension
            nn.Softmax(dim=-1)  # Ensures that the sum of two weights for each feature is 1
        )

    def forward(self, x):
        return self.gate(x).chunk(2, dim=-1)
        
        


class TwoStageAttentionLayerCrossSegmentsMultiMTS(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''
    def __init__(self, seg_num, factor, d_model, n_heads,sparsity, d_ff=None, dropout=0.0):
        super(TwoStageAttentionLayerCrossSegmentsMultiMTS, self).__init__()
        d_ff = d_ff or 4 * d_model
        #self.dim_sender = AttentionLayerCrossSegments(d_model, n_heads, dropout=dropout, return_attention=True)
        self.dim_senders = nn.ModuleList([AttentionLayerCrossSegments(d_model, n_heads, sparsity, dropout=dropout, return_attention=True) for _ in range(384)])
        
        self.time_attention = AttentionLayer(d_model, n_heads,52, dropout = dropout,return_attention=False)
        self.projection_layer = nn.Linear(2 * d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.norm31 = nn.LayerNorm(d_model)
        self.norm41 = nn.LayerNorm(d_model)

        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(384)])
        self.norm31s = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(384)])
        self.norm41s = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(384)])
        
        self.MLP2s = nn.ModuleList([nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)) for _ in range(384)])
        
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                nn.GELU(),
                                nn.Linear(d_ff, d_model))
        self.attention_details = [] 
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.gating_layer = GatingLayer(d_model) 
        
    def store_attn(self, i,att1, att2):
        self.attention_details.append({
            'pair_index': i,
            'att-A-B': att1.detach(),  # Detach tensors for storage
            'att-B-A': att2.detach(),
        })

    def get_attn(self):
        return self.attention_details

    def reset_attention_across_channels_details(self):
        self.attention_details = []

    def forward(self, x):
        # input shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
        batch = x.shape[0]
        ts_d = x.shape[1]
        seg_num = x.shape[2]
        d_model = x.shape[3]

        x_copy = x
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc = self.time_attention(
            time_in, time_in, time_in
        )
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)
        
        time_out = rearrange(dim_in, '(b ts_d) seg_num d_model -> b ts_d seg_num d_model', b = batch,ts_d=ts_d)
        x_copy = time_out
        # Reshape input to pair the time series"""
        # [batch_size, Data_dim(D), Seg_num(L), d_model] -> [batch_size, num_pairs, ts_d_per_pair, Seg_num(L), d_model]
        num_pairs = ts_d // 2
        ts_d_per_pair = 2
        x_copy = x_copy.view(batch, num_pairs, ts_d_per_pair, seg_num, d_model)

        # Apply MSA to each pair independently
        output = []
        for i in range(num_pairs):
            x_pair = x_copy[:, i]  # [batch_size, ts_d_per_pair, Seg_num(L), d_model]
            x_pair = x_pair.reshape(batch, seg_num * ts_d_per_pair, d_model)  # [batch_size, Seg_num(L) * ts_d_per_pair, d_model]

            # Cross dimension segment-segment
            dim_enc1, att1, att2 = self.dim_senders[i](x_pair, x_pair, x_pair, seg_num)
            self.store_attn(i,att1, att2)

            dim_enc1 = x_pair + self.dropouts[i](dim_enc1)
            #dim_enc1 = self.dropout(dim_enc1)
            dim_enc1 = self.norm31s[i](dim_enc1)
            dim_enc1 = dim_enc1 + self.dropouts[i](self.MLP2s[i](dim_enc1))
            dim_enc1 = self.norm41s[i](dim_enc1)

            output.append(dim_enc1)

        # Stack the output for each pair
        # [num_pairs, batch_size, ts_d_per_pair, Seg_num(L), d_model] -> [batch_size, num_pairs, ts_d_per_pair, Seg_num(L), d_model]
        output = torch.stack(output, dim=1)

        # Reshape the output to [batch_size, Data_dim(D), Seg_num(L), d_model]
        cross_out = output.reshape(batch, ts_d, seg_num, d_model)


        return cross_out
            