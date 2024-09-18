import ruptures as rpt


def get_breaks(signal, num_bkps, model_type='l2'):
    model = model_type
    alg = rpt.Dynp(model=model).fit(signal)
    my_bkps = alg.predict(num_bkps)
    return my_bkps
