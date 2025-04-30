from ..utils import importer


MODULE = "keypoint_diffuser.models"


def get_option_setter(model_name):
    return importer.get_option_setter(MODULE, model_name)


def get_model(model_name):
    return importer.get_model(MODULE, model_name)
