import torch
import torch.nn.functional as F
import math

def register_attention_control_new(model, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
            x = hidden_states
            context = encoder_hidden_states
            mask = attention_mask
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)

            q = self.head_to_batch_dim(q)
            k = self.head_to_batch_dim(k)
            v = self.head_to_batch_dim(v)

            q_new, k_new = controller(q, k, is_cross, place_in_unet)

            q_new = q_new.view(batch_size, h, q_new.shape[-2], q_new.shape[-1])
            k_new = k_new.view(batch_size, h, k_new.shape[-2], k_new.shape[-1])
            v = v.view(batch_size, h, v.shape[-2], v.shape[-1])
            
            if is_cross or attention_mask == None:
                mask = None
            else:
                scale_fac = math.sqrt(attention_mask.size(-1) * attention_mask.size(-2) // k_new.size(-2))
                mask = F.interpolate(mask, scale_factor = 1 / scale_fac)
                mask = mask.view(1, 1, -1)
                mask = torch.ger(mask.squeeze()/-10000, mask.squeeze() / -10000) * -1e6
            # todo: For cross, attn_mask = None ; For self, attn_mask = downsample(original_image_mask)
            out = F.scaled_dot_product_attention(
                q_new, k_new, v, attn_mask=mask, dropout_p=0.0, is_causal=False, scale = self.scale)
            out = out.transpose(1, 2).reshape(batch_size, -1, h * out.shape[-1])
            
            return to_out(out)

        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        # print(net_.__class__.__name__)
        if net_.__class__.__name__ == 'Attention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    # print(sub_nets)
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count

