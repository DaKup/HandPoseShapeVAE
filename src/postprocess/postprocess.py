import os
import sys

import itk
import numpy as np


def anisotropic_diffusion(depth_frame, num_iterations: int = 25, conductance: float = 0.8, timestep: float = 0.125):

    itk_input_image = itk.GetImageViewFromArray(depth_frame)

    InputImageType = itk.Image[itk.F, 2]
    OutputImageType = itk.Image[itk.F, 2]

    FilterType = itk.GradientAnisotropicDiffusionImageFilter[OutputImageType, OutputImageType]
    gradientfilter = FilterType.New()
    gradientfilter.SetInput(itk_input_image)
    gradientfilter.SetNumberOfIterations(num_iterations)
    gradientfilter.SetTimeStep(timestep)
    gradientfilter.SetConductanceParameter(conductance)

    return itk.GetArrayFromImage(gradientfilter.GetOutput())


def postprocess_frame(depth_frame, num_iterations: int = 25, conductance: float = 0.8, timestep: float = 0.125):

    return anisotropic_diffusion(depth_frame, num_iterations, conductance, timestep)


def postprocess_batch(depth_batch, num_iterations: int = 25, conductance: float = 0.8, timestep: float = 0.125):

    return np.array(list(postprocess_frame(frame, num_iterations, conductance, timestep) for frame in depth_batch), depth_batch.dtype)
