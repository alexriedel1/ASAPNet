import os.path

from PIL import Image

from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """
    def initialize(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        #BaseDataset.__init__(self, opt)
        self.opt = opt
        self.dir_AB = os.path.join(opt.dataroot,
                                   opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB))  # get image paths

        assert (self.opt.load_size >= self.opt.crop_size
                )  # crop_size should be smaller than the size of loaded image
        self.input_nc = 3
        self.output_nc = 3
        self.cache = {}

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')

        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt,
                                    transform_params,
                                    grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt,
                                    transform_params,
                                    grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)
        return {'label': A, 'image': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        if self.opt.max_dataset_size == -1:
            return len(self.AB_paths)
        else:
            return self.opt.max_dataset_size