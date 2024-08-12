import numpy as np
import albumentations as A
import imgaug.augmenters as iaa

class GenWithAlbumentation:

    @classmethod
    def _with_random_rain(cls):
        transform = A.RandomRain(
            slant_lower=-5,  # int | None
            slant_upper=5,  # int | None
            drop_length=10,  # int
            drop_width=1,  # int
            drop_color=(200, 200, 200),  # tuple[int, int, int]
            blur_value=4,  # int
            brightness_coefficient=0.7,  # float
            rain_type="drizzle",  # RainMode | None
            always_apply=None,  # bool | None
            p=1.0,  # float
        )
        return transform
    
    @classmethod
    def _with_random_fog(cls):
        transform = A.RandomFog(
            fog_coef_lower=0.2,  # float | None
            fog_coef_upper=0.6,  # float | None
            alpha_coef=0.09,  # float
            always_apply=None,  # bool | None
            p=1.0,  # float
        )
        
        return transform

    @classmethod
    def _with_random_fog_iaa(cls):
        transform = iaa.Sequential([
            iaa.Fog(),
            iaa.Clouds()
        ])
        
        return transform
        
    
    @classmethod
    def _with_random_SunFlare(cls):
        transform = A.RandomSunFlare(
            flare_roi=(0, 0, 1, 0.5),  # tuple[float, float, float, float]
            angle_lower=0,  # float | None
            angle_upper=1,  # float | None
            num_flare_circles_lower=10,  # int | None
            num_flare_circles_upper=20,  # int | None
            src_radius=200,  # int
            src_color=(255, 255, 255),  # tuple[int, ...]
            always_apply=None,  # bool | None
            p=1.0,  # float
        )
        
        return transform
    
    @classmethod
    def _with_random_BrightnessContrast(cls):
        transform = A.RandomBrightnessContrast(
            brightness_limit=(-0.2, -0.1),  # ScaleFloatType
            contrast_limit=(-0.2, 0.2),  # ScaleFloatType
            brightness_by_max=True,  # bool
            always_apply=None,  # bool | None
            p=1.0,  # float
        )
        return transform
    
    @classmethod
    def _with_PixelDropout(cls):
        transform = A.PixelDropout(
            dropout_prob=0.01,  # float
            per_channel=False,  # bool
            drop_value=0,  # ScaleFloatType | None
            mask_drop_value=None,  # ScaleFloatType | None
            always_apply=None,  # bool | None
            p=1.0,  # float
        )
        
        return transform
    
    @classmethod
    def _with_GlassBlur(cls):
        transform = A.GlassBlur(
            sigma=0.5,  # float
            max_delta=2,  # int
            iterations=2,  # int
            mode="fast",  # Literal['fast', 'exact']
            always_apply=None,  # bool | None
            p=1.0,  # float
        )
        return transform
    
    @classmethod
    def _with_Blur(cls):
        transform = A.OneOf([
            A.MotionBlur(
                blur_limit=9,  # ScaleIntType
                allow_shifted=True,  # bool
                always_apply=None,  # bool | None
                p=1.0,  # float
            ),
            A.GlassBlur(
                sigma=0.2,  # float
                max_delta=2,  # int
                iterations=2,  # int
                mode="fast",  # Literal['fast', 'exact']
                always_apply=None,  # bool | None
                p=1.0,  # float
            ),
            A.Blur(
                blur_limit=7,  # ScaleIntType
                p=1.0,  # float
                always_apply=None,  # bool | None
            ),
            A.GaussianBlur(
                blur_limit=(5, 7),  # ScaleIntType
                sigma_limit=0,  # ScaleFloatType
                always_apply=None,  # bool | None
                p=1.0,  # float
            ),
            A.ZoomBlur(
                max_factor=(1, 1.09),  # ScaleFloatType
                step_factor=(0.01, 0.02),  # ScaleFloatType
                always_apply=None,  # bool | None
                p=1.0,  # float
            )
        ], p=1)
        
        return transform
    
    @classmethod
    def _with_Posterize(cls):
        transform = A.Posterize(
            num_bits=4,  # int | tuple[int, int] | tuple[int, int, int]
            always_apply=None,  # bool | None
            p=1.0,  # float
        )
        return transform
    
    @classmethod
    def compose_transformation(cls, image):
        transforms = {
            # 'random_rain': cls._with_random_rain(),
            'random_fog': cls._with_random_fog(),
            'random_brightness_contrast': cls._with_random_BrightnessContrast(),
        }
        aug = np.random.choice(list(transforms.keys()))
        transform = transforms[aug]
        transformed = transform(image=image)['image']
        
        return aug, transformed