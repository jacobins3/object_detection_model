class SamLoRAConfig:
    try:
        import torch
        import os
        from pathlib import Path
    except:
        pass

    def on_batch_begin(self, learn, model_input_batch, model_target_batch, **kwargs):
        """
        Function to transform the input data and the targets in accordance to the model for training.
        """

        # model_input: [BxCxHxW], Normalized using ImageNet stats
        model_input = model_input_batch
        # model_target: [BxHxW], values in 0-n, n is #classes
        model_target = (model_target_batch).squeeze(dim=1)
        return model_input, model_target

    def transform_input(self, xb):
        """
        Function to transform the inputs for inferencing.
        """
        model_input = xb

        return model_input

    def transform_input_multispectral(self, xb):
        """
        Function to transform the multispectral inputs for inferencing.
        """
        model_input = xb
        return model_input

    def get_model(self, data, backbone=None, **kwargs):
        """
        Function used to define the model architecture.
        """
        from arcgis.learn.models._sam_lora_utils import LoRA_Sam, sam_model_registry
        from arcgis.learn._utils.utils import compare_checksum

        self.num_classes = data.c - 1  # 0-background, 1-buildings
        img_size = data.chip_size
        vit_name = backbone if type(backbone) is str else backbone.__name__
        ckpt = {
            "vit_b": "sam_vit_b_01ec64.pth",
            "vit_l": "sam_vit_l_0b3195.pth",
            "vit_h": "sam_vit_h_4b8939.pth",
        }
        file_checksum = {
            "vit_b": 1442441403,
            "vit_l": 1653302572,
            "vit_h": 2791087151,
        }

        rank = 4

        load_sam_weights = kwargs.get("load_sam_weights", False)
        if load_sam_weights:
            # Download (if required) SAM pretrained weights
            weights_path = self.os.path.join(self.Path.home(), ".cache", "weights")
            weights_file = self.os.path.join(weights_path, ckpt[vit_name])

            # Delete incomplete/corrupt downloads
            if self.os.path.exists(weights_file) and not compare_checksum(
                weights_file, file_checksum[vit_name]
            ):
                self.os.remove(weights_file)

            if not self.os.path.exists(weights_file):
                if not self.os.path.exists(weights_path):
                    self.os.makedirs(weights_path)
                try:
                    self.download_sam_weights(weights_file, ckpt[vit_name])
                except Exception as e:
                    print(e)
                    print(
                        "[INFO] Can't download SAM pretrained weights.\nProceeding without pretrained weights."
                    )
                    weights_file = None
        else:
            # Not to load sam weights if dlpk is to be used
            weights_file = None

        # register model
        sam, img_embedding_size = sam_model_registry[vit_name](
            image_size=img_size,
            num_classes=self.num_classes,
            checkpoint=weights_file,
        )
        model = LoRA_Sam(sam, rank).cuda()

        multimask_output = True
        low_res = img_embedding_size * 4

        self.model = model
        return model

    def loss(self, model_output, *model_target):
        """
        Function to define the loss calculations.
        """
        from torch.nn.modules.loss import CrossEntropyLoss

        ce_loss = CrossEntropyLoss()
        dice_weight = 0.8
        label = model_target[0]  # model_target is a single element tuple

        # Not using the Dice Loss from SAMed
        # Calculating loss by using image size output masks and target labels.
        # from arcgis.learn.models._sam_lora_utils import DiceLoss

        # loss_ce = ce_loss(model_output, label[:].long())
        # dice_loss = DiceLoss(self.num_classes + 1)
        # loss_dice = dice_loss(model_output, label, softmax=True)
        # loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
        # final_loss = loss

        # Using common Dice Loss
        from arcgis.learn._utils.segmentation_loss_functions import DiceLoss

        dice_loss = DiceLoss(
            ce_loss,
            dice_weight,
            weighted_dice=False,
            dice_average="micro",
        )
        final_loss = dice_loss(model_output, label)

        return final_loss

    def post_process(self, pred, thres, thinning=True, prob_raster=False):
        """
        Fuction to post process the output of the model in validation/infrencing mode.
        """
        if prob_raster:
            return pred
        else:
            output_masks = self.torch.argmax(
                self.torch.softmax(pred, dim=1), dim=1, keepdim=True
            )
            post_processed_pred = output_masks
        return post_processed_pred

    def download_sam_weights(self, weights_path, ckpt):
        from urllib.request import urlretrieve

        url = f"https://dl.fbaipublicfiles.com/segment_anything/{ckpt}"
        urlretrieve(url, weights_path)
