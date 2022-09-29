import logging

from torch.optim import SGD, Adam, ASGD, Adamax, Adadelta, Adagrad, RMSprop

logger = logging.getLogger("ptsemseg")

key2opt = {
    "sgd": SGD,
    "adam": Adam,
    "asgd": ASGD,
    "adamax": Adamax,
    "adadelta": Adadelta,
    "adagrad": Adagrad,
    "rmsprop": RMSprop,
}


def get_optimizer(cfg_opt):
    if cfg_opt is None:
        logger.info("Using SGD optimizer")
        return SGD

    else:
        opt_name = cfg_opt["name"]
        if opt_name not in key2opt:
            raise NotImplementedError("Optimizer {} not implemented".format(opt_name))

        logger.info("Using {} optimizer".format(opt_name))
        return key2opt[opt_name]
