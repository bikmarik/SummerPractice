class NullWriter:
    """
    No-op writer used for local smoke tests.
    Real training config uses WandB as required by the homework.
    """

    def __init__(self, *args, **kwargs):
        self.step = 0

    def set_step(self, step, mode="train"):
        self.step = step

    def add_scalar(self, *args, **kwargs):
        pass

    def add_scalars(self, *args, **kwargs):
        pass

    def add_checkpoint(self, *args, **kwargs):
        pass

    def add_image(self, *args, **kwargs):
        pass

    def add_audio(self, *args, **kwargs):
        pass

    def add_text(self, *args, **kwargs):
        pass

    def add_histogram(self, *args, **kwargs):
        pass

    def add_table(self, *args, **kwargs):
        pass
