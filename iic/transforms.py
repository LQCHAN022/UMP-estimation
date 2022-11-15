import torchvisision


class IntentityTransformIIC(list):
    def __init__(self):
        """
        Defines the base of tranformations (denoted g in TODO).
      A identity transform
      """

    pass

    def forward_transform(self, x):
        return x

    def backward_transform(self, x):
        return x

    def copy(self):
        return IntentityTransformIIC()


class RandomVFlipTransformIIC(IntentityTransformIIC):

    def __init__(self, p=0.5):
        """
        Performs a random vertical flip of the current batch, by
        """
        super().__init__()
        self.p = p
        self.was_flipped = None

    def forward_transform(self, x):
        if self.p > torch.rand(1):
            self.was_flipped = True
            return torchvision.transforms.functional.vflip(x)
        else:
            self.was_flipped = False
            return x

    def backward_transform(self, x):
        assert self.was_flipped is not None, "First Forward Pass needs to be executed"
        if self.was_flipped:
            return torchvision.transforms.functional.vflip(x)
        else:
            return x

    def copy(self):
        trans = RandomVFlipTransformIIC()
        trans.p = self.p
        trans.was_flipped = self.was_flipped
        return trans


class RandomHFlipTransformIIC(IntentityTransformIIC):
    def __init__(self, p=0.5):
        """
        Performs a random horizontal flip with propability p
        :param p:
        """
        super().__init__()
        self.p = p
        self.was_flipped = None

    def forward_transform(self, x):
        if self.p > torch.rand(1):
            self.was_flipped = True
            return torchvision.transforms.functional.hflip(x)
        else:
            self.was_flipped = False
            return x

    def backward_transform(self, x):
        assert self.was_flipped is not None, "First Forward Pass needs to be executed"
        if self.was_flipped:
            return torchvision.transforms.functional.hflip(x)
        else:
            return x

    def copy(self):
        trans = RandomHFlipTransformIIC()
        trans.p = self.p
        trans.was_flipped = self.was_flipped
        return trans


class ComposeTransformIIC(IntentityTransformIIC):
    def __init__(self, transforms):
        self.transforms = transforms

    def forward_transform(self, x):
        for transform in self.transforms:
            x = transform.forward_transform(x)
        return x

    def backward_transform(self, x):
        for transform in reversed(self.transforms):
            x = transform.backward_transform(x)
        return x

    def copy(self):
        transforms_out = []
        for transform in self.transforms:
            transforms_out.append(transform.copy())
        return ComposeTransformIIC(transforms_out)
