def freeze_base_model_for_lora(model):
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model
