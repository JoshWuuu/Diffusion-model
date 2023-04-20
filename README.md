# Diffusion-model
## Summary
It is a side project in 2023. The project is a Computer Vision topic. The languages and relevent packages are **Python - Pytorch**. The project aims to generate image using diffusion model. 
## Data
torchvision.datasets.StanfordCars, [link](http://ai.stanford.edu/~jkrause/cars/car_dataset.html). The cars dataset contain 16,185 images of 196 classes of cars. 
## Model
* Forward diffusion process: noise generation with beta value. Each diffusion is the original image adding with noise.
$$q(x_{1:T}|x_{0}) = \prod_{t=1}^T q(x_t|x_{t-1})$$ 
$$q(x_t|x_{t-1}) = N(x_t;\sqrt{1-\beta_t}x_{t-1}, \beta_tI)$$
$$X_t = \sqrt{(1 - \beta_t)} * X_{t-1} + \beta_t * noise $$ 
$$noise = N(0, 1)$$
* Backward denoising process: A Unet, containing downward, upward and skip connection between two paths. The noise is generated with gaussian distribution, then is feed into Unet with timestamp of the noise to predict the noise within the images.
$$p_\theta(x_T) = N(x_t;0, I)$$
$$q(x_{0:T}) = p(x_T)\prod_{t=1}^T p_\theta(x_{t-1}|x_{t})$$ 

## Result
<figure>
  <img 
  src="diffusion_sample_img.png" 
  alt="Results of sklearn models" 
  width="1300" height="150">
</figure>
The leftmost image is the denoising images at timestamp 0. And the rightmost image is the denoising images at timestamp 1000.

## Reference
https://github.com/lucidrains/denoising-diffusion-pytorch
