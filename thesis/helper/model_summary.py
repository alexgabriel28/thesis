def summary(model):
  params = list(model.named_parameters())
  line_sep = "-------------------------------------------------------------------------------------------------"
  print(line_sep)
  print("{:>30}  {:>30} {:>30}".format("Layer", "Shape", "No. Parameters"))
  print(line_sep)
  for elem in params:
    layer = elem[0]
    shape = list(elem[1].size())
    count = torch.tensor(elem[1].size()).prod().item()
    print("{:>30}  {:>30} {:>30}".format(layer, str(shape), str(count)))
  print(line_sep)
  sum_params = sum([param.nelement() for param in model.parameters()])
  print("Total Parameters:", sum_params)
  train_params = sum(params.numel() for params in model.parameters() if params.requires_grad)
  print("Trainable Parameters:", train_params)
  print("Non-Trainable Parameters:", sum_params - train_params)
  print(line_sep)