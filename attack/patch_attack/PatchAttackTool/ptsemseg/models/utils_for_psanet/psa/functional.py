from ptsemseg.models.utils_for_psanet.psa import functions


def psa_mask(input, psa_type=0, mask_H_=None, mask_W_=None):
    return functions.psamask(input, psa_type, mask_H_, mask_W_)
