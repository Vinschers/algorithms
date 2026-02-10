from torchvision import transforms


class SimCLRTransform:
    def __init__(self, img_size, s=1):

        color_jitter = transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)

        kernel_size = int(0.1 * img_size)
        if kernel_size % 2 == 0:
            kernel_size += 1

        self.data_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                rnd_color_jitter,
                rnd_gray,
                transforms.GaussianBlur(kernel_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # values from ImageNet, so that we can use algorithms pre-trained on ImageNet
            ]
        )

    def __call__(self, x):
        # it outputs a tuple, namely 2 views (augmentations) for the same image
        return self.data_transform(x), self.data_transform(x)
