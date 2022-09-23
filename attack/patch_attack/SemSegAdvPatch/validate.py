import yaml
import torch
import argparse
import timeit
import numpy as np

from torch.utils import data
import os


from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.metrics import runningScore
from ptsemseg.utils import convert_state_dict
from ptsemseg.utils import get_model_state

torch.backends.cudnn.benchmark = True


def validate(cfg, args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(cfg["device"]["gpu"])

    # Setup Dataloader
    data_loader = get_loader(cfg["data"]["dataset"])
    data_path = cfg["data"]["path"]

    loader = data_loader(
        data_path,
        split=cfg["data"]["val_split"],
        is_transform=True,
        version= cfg["data"]["version"],
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
        img_norm = cfg["data"]["img_norm"],
        bgr = cfg["data"]["bgr"], 
        std_version = cfg["data"]["std_version"],
        bottom_crop = 0
    )

    n_classes = loader.n_classes

    batch_size = 2 #cfg["training"]["batch_size"]
    valloader = data.DataLoader(loader, batch_size=batch_size, num_workers=1)
    running_metrics = runningScore(n_classes)

    # Setup Model
    model_file_name = os.path.split(cfg["model"]["path"])[1]
    model_name = model_file_name[: model_file_name.find("_")]
    print(model_name)
    model_dict = {"arch": cfg["model"]["arch"]} #model_dict = {"arch": model_name}
    model = get_model(model_dict, n_classes, version=cfg["data"]["dataset"])
    state = torch.load(cfg["model"]["path"], map_location = 'cpu')
    state = get_model_state(state, model_name)
    model.load_state_dict(state)
    
    model.to(device)
    model.eval()

    for i, (images, labels) in enumerate(valloader):
        start_time = timeit.default_timer()

        with torch.no_grad():

            images = images.to(device)

            if loader.bottom_crop > 0:
                images, labels = loader.crop_image(images, labels)

            if args.eval_flip:
                outputs = model(images)

                # Flip images in numpy (not support in tensor)
                outputs = outputs.data.cpu().numpy()
                flipped_images = np.copy(images.data.cpu().numpy()[:, :, :, ::-1])
                flipped_images = torch.from_numpy(flipped_images).float().to(device)
                outputs_flipped = model(flipped_images)
                outputs_flipped = outputs_flipped.data.cpu().numpy()
                outputs = (outputs + outputs_flipped[:, :, :, ::-1]) / 2.0

                pred = np.argmax(outputs, axis=1)
            else:
                outputs = model(images)
                pred = outputs.data.max(1)[1].cpu().numpy()
            
            if isinstance(labels, list):
                labels = labels[0]
            gt = labels.numpy()

            if args.measure_time:
                elapsed_time = timeit.default_timer() - start_time
                print(
                    "Inference time \
                    (iter {0:5d}): {1:3.5f} fps".format(
                        i + 1, pred.shape[0] / elapsed_time
                    )
                )
            running_metrics.update(gt, pred)

    score, class_iou = running_metrics.get_scores()

    for k, v in score.items():
        print(k, v)

    for i in range(n_classes):
        print(i, class_iou[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparams")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/fcn8s_pascal.yml",
        help="Config file to be used",
    )
    parser.add_argument(
        "--model_path",
        nargs="?",
        type=str,
        default="fcn8s_pascal_1_26.pkl",
        help="Path to the saved model",
    )
    parser.add_argument(
        "--eval_flip",
        dest="eval_flip",
        action="store_true",
        help="Enable evaluation with flipped image |\
                              True by default",
    )
    parser.add_argument(
        "--no-eval_flip",
        dest="eval_flip",
        action="store_false",
        help="Disable evaluation with flipped image |\
                              True by default",
    )
    parser.set_defaults(eval_flip=False)

    parser.add_argument(
        "--measure_time",
        dest="measure_time",
        action="store_true",
        help="Enable evaluation with time (fps) measurement |\
                              True by default",
    )
    parser.add_argument(
        "--no-measure_time",
        dest="measure_time",
        action="store_false",
        help="Disable evaluation with time (fps) measurement |\
                              True by default",
    )
    parser.set_defaults(measure_time=False)

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    validate(cfg, args)
