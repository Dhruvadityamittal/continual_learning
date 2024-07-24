def freeze_weights(model):
    i = 0
    # Set Batch_norm running stats to be frozen
    # or it will lead to bad results
    # Only freezing ConV layers for now
    # http://blog.datumbox.com/the-batch-normalization-layer-of-keras-is-broken/
    for name, param in model.named_parameters():
        if name.split(".")[0] == "feature_extractor":
            param.requires_grad = False
            i += 1

    batch_norm_freeze(model)
    print("Weights being frozen: %d" % i)


def batch_norm_freeze(model):
    """
    Freeze batch norm running stats.
    """

    bns_being_frozen = 0
    bns_not_frozen = 0

    for name, module in model.named_modules():
        classname = module.__class__.__name__
        if classname.find("BatchNorm1d") != -1:
            if "feature_extractor" in name.split("."):
                module.eval()
                bns_being_frozen += 1
            else:
                bns_not_frozen += 1

    assert bns_being_frozen > 1, "BatchNorms being frozen: %d" % bns_being_frozen
    assert bns_not_frozen <= 2 and bns_not_frozen >= 1, (
        "BatchNorms not being frozen: %d" % bns_not_frozen
    )
    print("BatchNorms being frozen: %d" % bns_being_frozen)
    print("BatchNorms not being frozen: %d" % bns_not_frozen)
