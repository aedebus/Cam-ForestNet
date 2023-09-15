import torch

from cams import *


class CAMModule:

    def set_up_cam(self, args):
        self.cam_dir = Path(args.save_path) / args.exp_name / CAM_DIR
        self.set_weights()
        self.set_up_hooks()

    def set_weights(self):
        """Normalize the gradients and aggregate them to 1 weight per fmap."""
        modules = dict(self.model.named_modules())
        if 'model.classifier' in modules:
            linear_module = modules['model.classifier']
        elif 'model.fc' in modules:
            linear_module = modules['model.fc']
        weights = dict(linear_module.named_parameters())['weight']
        self.weights = normalize_gradients(weights)

    def set_up_hooks(self):
        """Set the fmaps and grads attributes."""
        cam_layer = MODEL2CAM_LAYER[self.model.__class__.__name__]
        named_modules = dict(self.model.named_modules())
        module = named_modules[cam_layer]

        def save_fmaps(_, __, fmaps):
            self.fmaps = fmaps.detach()

        module.register_forward_hook(save_fmaps)

    def get_cam(self):
        batch, num_channels, height, width = self.fmaps.size()  # (b, 2048, 4, 4)
        self.fmaps = self.fmaps.reshape(
            num_channels, batch * height * width)
        # (num_classes, num_channel)*(num_channels, b*h*w)
        cam = torch.mm(self.weights.cuda(), self.fmaps.cuda())
        cam = cam.reshape(
            cam.size()[0],
            batch,
            height,
            width)  # (num_classes, b, h, w)
        cam = torch.transpose(cam, 0, 1)  # (b, num_classes, h, w)
        # Clip negative values to 0
        cam = torch.clamp(cam, min=0, max=float('inf'))
        # Normalize CAM values (per CAM in the batch)
        # Must append dimensions so that the subtract broadcast works.
        cam -= cam.min(dim=2)[0].min(dim=2)[0][:, :, None, None]
        cam /= (cam.max(dim=2)[0].max(dim=2)[0][:, :, None, None] + 1e-7)

        return cam.cpu().numpy()

    def write_cams(self, x, y, logits, idx):
        cams = self.get_cam()
        test_dataset = self.test_dataloader().dataset
        zoomed_regions = self.hparams['resize'] == "zoomed"
        write_batch_cams(batch_x=x['image'],
                         batch_y=y,
                         batch_y_hat=logits,
                         batch_cams=cams,
                         idxs=idx,
                         save_dir=self.cam_dir,
                         dataset=test_dataset,
                         zoomed_regions=zoomed_regions)
