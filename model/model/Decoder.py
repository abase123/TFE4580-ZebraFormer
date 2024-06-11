#torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
#Libaries for calculation and processing
from einops import rearrange


from .TSA import AttentionLayerCrossSegments, TwoStageAttentionLayerCrossSegments, AttentionLayer, TwoStageAttentionLayerCrossSegmentsMultiMTS


class DecoderLayer(nn.Module):
    '''
    The decoder layer of Crossformer, each layer will make a prediction at its scale
    '''
    def __init__(self, seg_len, d_model, n_heads,sparsity, d_ff=None, dropout=0.1, out_seg_num = 10, factor = 10):
        super(DecoderLayer, self).__init__()
        self.self_attention = TwoStageAttentionLayerCrossSegmentsMultiMTS(out_seg_num, factor, d_model, n_heads, 104, \
                                d_ff, dropout)    
        self.cross_attention = AttentionLayerCrossSegments(d_model, n_heads, sparsity=1, dropout = dropout,cross_flag=True,return_attention=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_model),
                                nn.GELU(),
                                nn.Linear(d_model, d_model))
        #self.linear_pred = nn.Linear(d_model, seg_len)
        self.linear_preds = nn.ModuleList([nn.Linear(d_model, seg_len) for _ in range(768)])

        self.dim_senders = nn.ModuleList([AttentionLayerCrossSegments(d_model, n_heads, 104, dropout=dropout, cross_flag=True, return_attention=True) for _ in range(384)])
        self.norm31s = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(768)])
        self.norm1s = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(768)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(768)])
        self.MLP2s = nn.ModuleList([nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)) for _ in range(768)])

    
        self.attention_details = []
        self.cross_attentions = nn.ModuleList([
            AttentionLayerCrossSegments(d_model, n_heads, sparsity=1, dropout=dropout, cross_flag=True, return_attention=True) 
            for _ in range(768)  # Assuming 768 is your ts_d count based on previous usage
        ])
 


    def forward(self, x, cross):
        '''
        x: the output of last decoder layer
        cross: the output of the corresponding encoder layer
        '''
        batch = x.shape[0]
        seg_num = x.shape[2]
        d_model = x.shape[3]
        ts_d = x.shape[1]
        num_pairs = ts_d // 2
    
        x = self.self_attention(x)

        #x = rearrange(x, 'b ts_d out_seg_num d_model -> (b ts_d) out_seg_num d_model')
        #cross = rearrange(cross, 'b ts_d in_seg_num d_model -> (b ts_d) in_seg_num d_model')

        #x = x.view(batch,num_pairs, self.ts_d_per_pair, seg_num, d_model)
        #cross = cross.view(batch,num_pairs, self.ts_d_per_pair, seg_num, d_model)

        """for i in range(num_pairs):
            dec_pair = x[:, i].reshape(batch, seg_num * self.ts_d_per_pair, d_model)
            enc_pair = cross[:, i].reshape(batch, seg_num * self.ts_d_per_pair, d_model)

            #Cross-attention where decoder attends to encoder
            attn_output  = self.dim_senders[i](dec_pair, enc_pair, enc_pair, seg_num)
            
            attn_output = dec_pair + self.dropouts[i](attn_output)
            attn_output = self.norm31s[i](attn_output)
            attn_output = attn_output + self.dropouts[i](self.MLP2s[i](attn_output))
            attn_output = self.norm31s[i](attn_output)

            output.append(attn_output.view(batch, self.ts_d_per_pair, seg_num, d_model))"""
        #x = rearrange(x, 'b ts_d out_seg_num d_model -> (b ts_d) out_seg_num d_model')
        #cross = rearrange(cross, 'b ts_d in_seg_num d_model -> (b ts_d) in_seg_num d_model')
        #x = x + self.dropout(tmp)
        #y = x = self.norm1(x)
        #y = self.MLP1(y)
        
        #dec_output = self.norm2(x+y)
        #output = torch.stack(output, dim=1)
        #output = output.reshape(batch, num_pairs * self.ts_d_per_pair, seg_num, d_model)
        #dec_output = self.norm2(x+y)

        #tmp = self.cross_attention(
        #    x, cross, cross, None
        #)
        
        dec_outputs = []
        for i in range(ts_d):
            segment_x = x[:, i, :, :]
            segment_cross = cross[:, i, :, :]
            tmp = self.cross_attentions[i](segment_x, segment_cross, segment_cross, None)
            segment_x = segment_x + self.dropouts[i](tmp)
            segment_x = self.norm1s[i](segment_x)
            segment_x = self.MLP2s[i](segment_x)
            segment_x = self.norm31s[i](segment_x + segment_x)

            dec_outputs.append(segment_x)

        dec_output = torch.stack(dec_outputs, dim=1)

        #dec_output = rearrange(dec_output, '(b ts_d) seg_dec_num d_model -> b ts_d seg_dec_num d_model', b = batch)

        #layer_predict = self.linear_pred(dec_output)
        outputs = [linear(dec_output[:, i, :, :]) for i, linear in enumerate(self.linear_preds)]

        layer_predict = torch.stack(outputs, dim=1)

        layer_predict = rearrange(layer_predict, 'b out_d seg_num seg_len -> b (out_d seg_num) seg_len')

        return dec_output, layer_predict

class Decoder(nn.Module):
    '''
    The decoder of Crossformer, making the final prediction by adding up predictions at each scale
    '''
    def __init__(self, seg_len, d_layers, d_model, n_heads, d_ff, dropout,\
                router=False, out_seg_num = 10, factor=10):
        super(Decoder, self).__init__()

        self.router = router
        self.sparsity_list = [9,9,9]
        self.decode_layers = nn.ModuleList()
        for i in range(d_layers):
            self.decode_layers.append(DecoderLayer(seg_len, d_model, n_heads,self.sparsity_list[i],d_ff, dropout, \
                                        out_seg_num, factor))

    def forward(self, x, cross):
        final_predict = None
        i = 0
      
        ts_d = x.shape[1]
        for layer in self.decode_layers:
            cross_enc = cross[i]
            x, layer_predict = layer(x,  cross_enc)
            if final_predict is None:
                final_predict = layer_predict
            else:
                final_predict = final_predict + layer_predict
            i += 1
        
        final_predict = rearrange(final_predict, 'b (out_d seg_num) seg_len -> b (seg_num seg_len) out_d', out_d = ts_d)

        return final_predict

