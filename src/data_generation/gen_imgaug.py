import numpy as np
import imgaug.augmenters as iaa


class RandomFog(iaa.CloudLayer):
    def __init__(self, seed=None, name=None, random_state="deprecated", deterministic="deprecated"):
        super(RandomFog, self).__init__(
            intensity_mean=(220, 255),
            intensity_freq_exponent=(-2.0, -1.5),
            intensity_coarse_scale=2,
            alpha_min=(0.7, 0.9),
            alpha_multiplier=0.3,
            alpha_size_px_max=(2, 4),
            alpha_freq_exponent=(-4.0, -2.0),
            sparsity=0.9,
            density_multiplier=(0.4, 0.6),
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic
        )
        

class RamdomClouds(iaa.meta.SomeOf):
    def __init__(self,
                seed=None, name=None,
                random_state="deprecated", deterministic="deprecated"):
        layers = [
            iaa.CloudLayer(
                intensity_mean=(196, 255),
                intensity_freq_exponent=(-2.5, -2.0),
                intensity_coarse_scale=10,
                alpha_min=0,
                alpha_multiplier=(0.25, 0.75),
                alpha_size_px_max=(2, 8),
                alpha_freq_exponent=(-2.5, -2.0),
                sparsity=(0.8, 1.0),
                density_multiplier=(0.5, 1.0),
                seed=seed,
                random_state=random_state,
                deterministic=deterministic
            ),
            iaa.CloudLayer(
                intensity_mean=(196, 255),
                intensity_freq_exponent=(-2.0, -1.0),
                intensity_coarse_scale=10,
                alpha_min=0,
                alpha_multiplier=(0.5, 1.0),
                alpha_size_px_max=(64, 128),
                alpha_freq_exponent=(-2.0, -1.0),
                sparsity=(1.0, 1.4),
                density_multiplier=(0.8, 1.5),
                seed=seed,
                random_state=random_state,
                deterministic=deterministic
            )
        ]

        super(RamdomClouds, self).__init__(
            (1, 2),
            children=layers,
            random_order=False,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class RandomRain(iaa.meta.SomeOf):
    def __init__(self, nb_iterations=(1, 3),
                drop_size=(0.01, 0.02),
                speed=(0.04, 0.20),
                seed=None, name=None,
                random_state="deprecated", deterministic="deprecated"):
        layer = iaa.RainLayer(
            density=(0.03, 0.14),
            density_uniformity=(0.8, 1.0),
            drop_size=drop_size,
            drop_size_uniformity=(0.2, 0.5),
            angle=(-15, 15),
            speed=speed,
            blur_sigma_fraction=(0.001, 0.001),
            seed=seed,
            random_state=random_state,
            deterministic=deterministic
        )

        super(RandomRain, self).__init__(
            nb_iterations,
            children=[layer.deepcopy() for _ in range(3)],
            random_order=False,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class GenWithImgAug:
    @classmethod
    def compose_transformation(cls, image):
        transforms = {
            'random_fog':RandomFog(),
            'random_cloulds': RamdomClouds(),
            'random_rain': RandomRain(),
        }
        aug = np.random.choice(list(transforms.keys()))
        transform = transforms[aug]
        transformed = transform(image=image)
        
        return aug, transformed