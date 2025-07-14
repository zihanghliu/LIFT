import copy

from s2ft import S2ColumnLinear, S2RowLinear

def only_optimize_s2_parameters(model):
    # Turn off the gradient of all the parameters except the S2 parameters
    for name, param in model.named_parameters():
        if "s2" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


# convert the linear layers in the MHA module to S2
def convert_mha_layer_to_s2(model, selected_parameters):
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    for i in range(model.config.num_hidden_layers):
        layer = model.model.layers[i]
        only_v = list(
            set(selected_parameters["v_proj"][i])
            - set(selected_parameters["o_proj"][i])
        )
        only_o = list(
            set(selected_parameters["o_proj"][i])
            - set(selected_parameters["v_proj"][i])
        )
        vo = list(
            set(selected_parameters["o_proj"][i])
            & set(selected_parameters["v_proj"][i])
        )
        order = only_v + vo + only_o
        for j in range(model.config.num_attention_heads):
            if j not in order:
                order.append(j)
        if len(only_v) + len(vo) > 0:
            module = layer.self_attn.v_proj
            checkpoint = copy.deepcopy(module.state_dict())
            layer.self_attn.v_proj = S2ColumnLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias,
                start=0,
                end=(len(only_v) + len(vo)) * head_dim,
                device=next(module.parameters()).device,
                dtype=next(module.parameters()).dtype,
            )
            layer.self_attn.v_proj.load_state_dict(checkpoint, strict=False)
            del module
            del checkpoint
        if len(only_o) + len(vo) > 0:
            module = layer.self_attn.o_proj
            checkpoint = copy.deepcopy(module.state_dict())
            layer.self_attn.o_proj = S2RowLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias,
                start=(len(only_v)) * head_dim,
                end=(len(only_v) + len(vo) + len(only_o)) * head_dim,
                device=next(module.parameters()).device,
                dtype=next(module.parameters()).dtype,
            )
            layer.self_attn.o_proj.load_state_dict(checkpoint, strict=False)
            del module
            del checkpoint

        q_weight = layer.self_attn.q_proj.weight.data
        q_weight = q_weight.reshape(
            model.config.num_key_value_heads, -1, q_weight.shape[-1]
        )
        layer.self_attn.q_proj.weight.data = q_weight[order, :, :].reshape(
            -1, q_weight.shape[-1]
        )
        k_weight = layer.self_attn.k_proj.weight.data
        k_weight = k_weight.reshape(
            model.config.num_key_value_heads, -1, k_weight.shape[-1]
        )
        layer.self_attn.k_proj.weight.data = k_weight[order, :, :].reshape(
            -1, k_weight.shape[-1]
        )
        v_weight = layer.self_attn.v_proj.weight.data
        v_weight = v_weight.reshape(
            model.config.num_key_value_heads, -1, v_weight.shape[-1]
        )
        layer.self_attn.v_proj.weight.data = v_weight[order, :, :].reshape(
            -1, v_weight.shape[-1]
        )
        o_weight = layer.self_attn.o_proj.weight.data
        o_weight = o_weight.reshape(
            o_weight.shape[0], model.config.num_attention_heads, -1
        )
        layer.self_attn.o_proj.weight.data = o_weight[:, order, :].reshape(
            o_weight.shape[0], -1
        )

        del v_weight, o_weight
    return model


# convert the linear layers in the FFN module to S2
def convert_ffn_layer_to_s2(model, selected_parameters):
    for i in range(model.config.num_hidden_layers):
        layer = model.model.layers[i]
        only_u = [
            j
            for j in selected_parameters["up_proj"][i]
            if j not in selected_parameters["down_proj"][i]
        ]
        only_d = [
            j
            for j in selected_parameters["down_proj"][i]
            if j not in selected_parameters["up_proj"][i]
        ]
        ud = [
            j
            for j in selected_parameters["up_proj"][i]
            if j in selected_parameters["down_proj"][i]
        ]
        order = only_u + ud + only_d
        for j in range(model.config.intermediate_size):
            if j not in order:
                order.append(j)
        if len(only_u) + len(ud) > 0:
            module = layer.mlp.up_proj
            checkpoint = copy.deepcopy(module.state_dict())
            layer.mlp.up_proj = S2ColumnLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias,
                start=0,
                end=(len(only_u) + len(ud)),
                device=next(module.parameters()).device,
                dtype=next(module.parameters()).dtype,
            )
            layer.mlp.up_proj.load_state_dict(checkpoint, strict=False)
            del module
            del checkpoint

        if len(ud) + len(only_d) > 0:
            module = layer.mlp.down_proj
            checkpoint = copy.deepcopy(module.state_dict())
            layer.mlp.down_proj = S2RowLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias,
                start=len(only_u),
                end=(len(only_u) + len(ud) + len(only_d)),
                device=next(module.parameters()).device,
                dtype=next(module.parameters()).dtype,
            )
            layer.mlp.down_proj.load_state_dict(checkpoint, strict=False)
            del module
            del checkpoint
        u_weight = layer.mlp.up_proj.weight.data
        layer.mlp.up_proj.weight.data = u_weight[order, :]
        g_weight = layer.mlp.gate_proj.weight.data
        layer.mlp.gate_proj.weight.data = g_weight[order, :]
        d_weight = layer.mlp.down_proj.weight.data
        layer.mlp.down_proj.weight.data = d_weight[:, order]


# convert the S2FT layer to linear layer
def convert_s2_to_linear_layer(model):
    for module in model.modules():
        if isinstance(module, S2ColumnLinear) or isinstance(module, S2RowLinear):
            module.fuse_s2_weight()
    return model