def set_parameter_requires_grad(model, require_grad = True):
    if require_grad == False:
        for param in model.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True