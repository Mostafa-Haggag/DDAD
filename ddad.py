from asyncio import constants
from typing import Any
import torch
from unet import *
from dataset import *
from visualize import *
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from anomaly_map import *
from metrics import *
from feature_extractor import *
from reconstruction import *
import cv2
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"


def makeVideoFromImageArray(output_filename, image_list):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # Defines the codec to be used for the video. XVID is a popular video codec.

    image_width = image_list[0].shape[1]
    image_height = image_list[0].shape[0]
    # Retrieves the width and height of the images from the first image in the list.
    # It assumes all images are of the same size.

    out = cv2.VideoWriter(filename=output_filename, fourcc=fourcc, fps=8,
                          frameSize=(image_width, image_height), isColor=True)
    # loop over the list of pictures
    for image_number, image in enumerate(image_list, 1):
        # Create a text overlay with the image number
        text = f"Frame {image_number}"
        font = cv2.FONT_HERSHEY_SIMPLEX # Chooses the font FONT_HERSHEY_SIMPLEX.
        font_scale = 0.5
        font_thickness = 1
        # Sets the font scale to 0.5 and thickness to 1.
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_x = 10
        text_y = 20

        # Add the text to the frame
        frame_with_text = cv2.cvtColor(image.copy(),cv2.COLOR_RGB2BGR)

        # Add the text to the frame
        # Uses cv2.putText to overlay the text on the image.
        # cv2.putText(frame_with_text, text, (text_x, text_y), font,
        #             font_scale, (255, 255, 255), font_thickness)

        out.write(frame_with_text)

    out.release()

class DDAD:
    def __init__(self, unet, config) -> None:
        self.test_dataset = Dataset_maker_test(
            root=config.data.data_dir,
            config=config,
        )
        self.testloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=config.data.test_batch_size,
            shuffle=False,
            num_workers=config.model.num_workers,
            drop_last=True,
        )
        self.unet = unet
        self.config = config
        self.reconstruction = Reconstruction(self.unet, self.config)
        self.transform = transforms.Compose([
                            transforms.CenterCrop((224)), 
                        ])

    def __call__(self) -> Any:
        feature_extractor = domain_adaptation(self.unet, self.config, fine_tune=False)
        feature_extractor.eval()
        
        labels_list = []
        predictions= []
        anomaly_map_list = []
        gt_list = []
        reconstructed_list = []
        reconstructed_list_numpy = []
        orginal_list_numpy = []

        forward_list = []
        gt_list_numpy = []
        anomaly_map_list_numpy = []
        reverse_transforms = transforms.Compose([
            transforms.Lambda(lambda t: (t + 1) / (2)),
            transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
            transforms.Lambda(lambda t: t * 255.),
            transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
        ])

        with torch.no_grad():
            for input, gt, labels in tqdm(self.testloader):
                input = input.to(self.config.model.device)
                x0 = self.reconstruction(input, input, self.config.model.w)[-1]
                anomaly_map = heat_map(x0, input, feature_extractor, self.config)

                anomaly_map = self.transform(anomaly_map)
                gt = self.transform(gt)

                forward_list.append(input)
                anomaly_map_list.append(anomaly_map)
                gt_list.append(gt)
                gt_list_numpy.append(gt.detach().cpu().numpy().astype(np.uint8))
                anomaly_map_list_numpy.append(anomaly_map.detach().cpu().numpy().astype(np.uint8))

                reconstructed_list.append(x0)
                reconstructed_list_numpy.append(reverse_transforms(x0.squeeze(0)))
                orginal_list_numpy.append(reverse_transforms(input.squeeze(0)))
                for pred, label in zip(anomaly_map, labels):
                    labels_list.append(0 if label == 'good' else 1)
                    predictions.append(torch.max(pred).item())

        makeVideoFromImageArray('video_constructed.avi', reconstructed_list_numpy)
        makeVideoFromImageArray('video_orginal.avi', orginal_list_numpy)

        gt_mask = (np.asarray(gt_list_numpy[:49])).astype(np.uint8)
        pred_flat = (np.asarray(anomaly_map_list_numpy[:49])).astype(np.uint8)
        gt_mask = gt_mask.flatten()
        pred_flat = pred_flat.flatten()
        conf_matrix = confusion_matrix(gt_mask, pred_flat)
        print(conf_matrix)
        metric = Metric(labels_list, predictions, anomaly_map_list, gt_list, self.config)
        metric.optimal_threshold()
        if self.config.metrics.auroc:
            print('AUROC: ({:.1f},{:.1f})'.format(metric.image_auroc() * 100, metric.pixel_auroc() * 100))
        if self.config.metrics.pro:
            print('PRO: {:.1f}'.format(metric.pixel_pro() * 100))
        if self.config.metrics.misclassifications:
            metric.miscalssified()
        reconstructed_list = torch.cat(reconstructed_list, dim=0)
        forward_list = torch.cat(forward_list, dim=0)
        anomaly_map_list = torch.cat(anomaly_map_list, dim=0)
        pred_mask = (anomaly_map_list > metric.threshold).float()
        gt_list = torch.cat(gt_list, dim=0)
        if not os.path.exists('results'):
                os.mkdir('results')
        if self.config.metrics.visualisation:
            visualize(forward_list, reconstructed_list, gt_list, pred_mask, anomaly_map_list, self.config.data.category)
