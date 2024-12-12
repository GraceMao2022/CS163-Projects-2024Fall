---
layout: post
comments: true
title: Post Template
author: UCLAdeepvision
date: 2024-01-01
---


> This block is a brief introduction of your project. You can put your abstract here or any headers you want the readers to know.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

---  

## Main Content
Your survey starts here. You can refer to the [source code](https://github.com/lilianweng/lil-log/tree/master/_posts) of [lil's blogs](https://lilianweng.github.io/lil-log/) for article structure ideas or Markdown syntax. We've provided a [sample post](https://ucladeepvision.github.io/CS188-Projects-2022Winter/2017/06/21/an-overview-of-deep-learning.html) from Lilian Weng and you can find the source code [here](https://raw.githubusercontent.com/UCLAdeepvision/CS188-Projects-2022Winter/main/_posts/2017-06-21-an-overview-of-deep-learning.md)

## Basic Syntax
### Image
Please create a folder with the name of your team id under /assets/images/, put all your images into the folder and reference the images in your main content.

You can add an image to your survey like this:
![YOLO]({{ '/assets/images/UCLAdeepvision/object_detection.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1. YOLO: An object detection method in computer vision* [1].

Please cite the image if it is taken from other people's work.


### Table
Here is an example for creating tables, including alignment syntax.

|             | column 1    |  column 2     |
| :---        |    :----:   |          ---: |
| row1        | Text        | Text          |
| row2        | Text        | Text          |



### Code Block
```
# This is a sample code block
import torch
print (torch.__version__)
```


### Formula
Please use latex to generate formulas, such as:

$$
\tilde{\mathbf{z}}^{(t)}_i = \frac{\alpha \tilde{\mathbf{z}}^{(t-1)}_i + (1-\alpha) \mathbf{z}_i}{1-\alpha^t}
$$

or you can write in-text formula $$y = wx + b$$.

### More Markdown Syntax
You can find more Markdown syntax at [this page](https://www.markdownguide.org/basic-syntax/).

## DALL-E V1: Variational Autoencoder + Transformer
The original DALL-E was based on the paper [“Zero-Shot Text-to-Image Generation”](https://arxiv.org/pdf/2102.12092) by Aditya Ramesh et al in 2021. It was developed by OpenAI.

In this paper, the authors propose a 12-billion parameter autoregressive transformer trained on 250 million image-text pairs for text to image generation, claiming that not limiting models to small training sets could possibly produce better results for specific text prompts. It also achieves zero-shot image generation, meaning that the model produces high quality images for labels that it was not specifically trained on.

### Method
The general idea for the model is to “train a transformer to autoregressively model the text and image tokens as a single stream of data” [ref]. 

DALL-E’s training procedure consists of two stages:
#### Stage 1: Learning the Visual Codebook
![DALLE-1]({{ '/assets/images/37/Dalle-1.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Figure N. DALL-E's dVAE autoencoder and decoder.”* [ref]

- Train a discrete variational autoencoder (dVAE) to downsample the 256x256 RGB input image into a 32x32 grid, with each grid unit possibly having 1 of 8192 possible values from the 8192 codebook vectors trained during this stage.
- The codebook vectors contain discrete latent indices and its distribution via the input image
- A dVAE decoder is also trained on this step to be able to regenerate images from the discrete codes
- This step is also important for mapping the input data to discrete codebook vectors in latent space, where each code allows the model to capture categorical information about the image.

#### Stage 2: Learning the Prior
![DALLE-2]({{ '/assets/images/37/Dalle-2.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Figure N. DALL-E's transformer.”* [ref]

- Concatenate up to 256 BPE-encoded (byte-pair encoded) text tokens with the 1024 image tokens and train an autoregressive transformer to model joint distribution over the tokens
- The transformer takes in text captions of training images and learns to produce the codebook vectors for one token, and then predicts the distribution for the next token until all 1024 tokens are produced.
- The dVAE decoder trained in stage 1 is used to generate the new image.

#### Loss Objective
The overall procedure maximizes the evidence lower bound (ELB) on the joint likelihood of the model distribution over images x, captions y, and tokens z for the encoded image.
![DALLE-3]({{ '/assets/images/37/Dalle-3.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Figure N. DALL-E's loss objective.”* [ref]

- q-phi is the distribution over image tokens given RGB input images
- p-theta is distribution over RGB images given image tokens
- p-psi is joint distribution over text and image tokens modeled by transformer

The goal of these stages is to reduce the amount of memory needed to train the model and to capture more low-frequency structure of the images rather than short-range dependencies between pixels that likelihood objectives tend to recognize.

### Results
![DALLE-4]({{ '/assets/images/37/Dalle-4.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Figure N. Text-to-image results from DALL-E v1.”* [ref]

Here are some results from the initial DALL-E paper. This model architecture has been overtaken by recent models like diffusion models, which even DALL-E 2 and 3 are using. However, it is still interesting to study different text-to-image generation techniques.

## Imagen: LLM + Diffusion Model
Imagen is a powerful text-to-image generation model developed by Google Brain which, at the time of its release, held the State of the Art performance on a variety of metrics, including the COCO FID benchmark. Key features of this innovative model are:  
- LLM for text encoder
- Efficient UNet backbone
- Classifier-free guidance with dynamic thresholding
- Noise augmentation and conditioning in cascading diffusion models

![Imagen Generation Examples](../assets/images/37/imagen_ex.png){: style="width: 680px; max-width: 100%;"}    
*Fig XX. Examples of Imagen Generation* [x]   

### Revisiting Diffusion Models
In Stable Diffusion, text conditioning is done via a separate transformer that encodes text prompts into the latent space of the diffusion model. These encoders are trained jointly with the diffusion backbone on pairs of images and their text captions. Thus, the text encoders are only exposed to a limited set of text data that is constrained to the space of image captioning and descriptors.  
  
Although this method of jointly training a text encoder and diffusion model works well in practice, Imagen finds that using pre-trained large-language models has much better performance. In fact, Imagen uses the largest language model available at the time (T5-XXL) which is a general language model. The advantages of this approach are twofold, there is no need to train a dedicated encoder, and general encoders learn much better semantic information.  
  
Not requiring a dedicated encoder makes the training process much easier, it is one less model that needs to be optimized. Using a separate and pre-trained model also allows training data to be directly encoded once, simplifying expensive computation that would have to be done for each training step. The Imagen authors state that text data is encoded one time, and that during training, the model can directly use (image, encoding) pairs rather than deriving encodings at every iteration from (image, text) pairs.  
  
In addition, using pre-trained LLM allows the generative model to capitalize on important semantic representations already encoded and learned from vast datasets. Simply put, pre-trained models have a much better understanding of language since they are trained on much larger and richer data than dedicated diffusion text encoders.   
  
The Imagen authors also report surprising results ‒ that the size of the language model has a direct correlation with better generative features. Scaling up the language model even has a much larger impact on generative capabilities than scaling the diffusion model itself.  

![Imagen Experiments](../assets/images/37/imagen_llmvsunet.png){: style="width: 500px; max-width: 100%;"}  
*Fig XX. Imagen Experiments with Varying LLM and UNet size* [x].  

These charts, from the Imagen paper, show the model capabilities for varying sizes of LLM text encoders (on the left) and UNet diffusion models (on the right). Evidently, increasing the size of the text encoder has a much more drastic improvement on evaluation metrics than increasing the size of the diffusion backbone. A bigger text encoder equals a better generative model.   
  
Imagen also adapts the diffusion backbone itself, introducing a new variant they call Efficient UNet. This version of the UNet model has considerably better memory efficiency, inference time, and convergence speed, being 2-3x faster than other UNets. Better sample quality and faster inference is essential for generative performance and training.  

### Further Optimizations
Imagen also introduces several other novel optimizations for the training process on top of revamping the diffusion backbone and text encoder. These optimizations include dynamic thresholding to enable aggressive classifier-free guidance as well as injecting noise augmentation and conditioning to boost upsampling quality.  
  
Classifier-free guidance is a technique used to improve the quality of text conditional training in diffusion models. Its counterpart, classifier guidance, has the effect of improving sample quality while reducing diversity in conditional diffusion. However, classifier guidance involves pre-trained models that themselves need to be optimized. Classifier-free guidance is an alternative that avoids pre-trained models by jointly training a single diffusion model on conditional and unconditional objectives by randomly dropping the condition during training. The strength of classifier-free guidance is also controlled by a weight parameter.  
  
Larger weights of classifier-free guidance improves text-image alignment but results in very saturated and unnatural images. Imagen addresses this issue with dynamic thresholding. In short, large weights on guidance pushes the values of the predicted image outside of the bounds of the training data (which are between -1, and 1). Runaway values lead to saturation. Dynamic thresholding constraints large values in the predicted image so that the resulting images are not overly saturated. The use of dynamic thresholding thus allows the use of aggressive guidance weights, which improve the model quality without leading to unnatural images.  
  
In addition, the Imagen architecture also implements cascading diffusion models. The first DM generates a 64x64 image, then the second upsamples to 256x256, and the final upsamples to 1024x1024. This cascading method results in very high fidelity images. Imagen further builds on this capability by introducing noise conditioning and augmentations in the cascading steps.   
  
Noise augmentation adds a random amount of noise to intermediate images between upsampling. Noise conditioning in the upsampling models means that the amount of noise added in the form of augmentations is provided as a further condition to the upsampling model.  
  
These techniques, taken together, boost the quality produced by the final Imagen model.  
  
### Overview of Imagen Architecture
First, text prompts are passed to a frozen text encoder that results in embeddings. These embeddings are inputs into a text-to-image diffusion model that outputs a 64x64 image.  
  
Then, the image is upsampled via two super-resolution models that create the final 1024x1024 image.   

![Imagen Architecture](../assets/images/37/imagen_arch.png){: style="width: 500px; max-width: 100%;"}  
*Fig XX. Imagen Architecture* [x].    

## Running Stable Diffusion via WebUI
Our group decided to use AUTOMATIC1111’s stable diffusion web UI to run pre-existing stable diffusion models. 

### Introduction to Stable Diffusion Web UI
Stable Diffusion Web UI is a web interface for Stable Diffusion, which provides traditional image to image and text to image capabilities. It also provides features like prompt matrices, where you can provide multiple prompts and a matrix of multiple images will be produced based on the combination of those prompts.

![Prompt Matrix]({{ '/assets/images/37/prompt-matrix.png' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Figure N. Prompt matrix. Several images all with the same seed are produced based on the combination of the prompts “a busy city street in a modern city|illustration|cinematic lighting,” where | indicates the separation of prompts.* [ref].

In addition, it provides an Attention feature where using () in the text prompt increases the model’s attention on those enclosed words, and using [] decreases it. Adding more () or [] around the same enclosed words increases the magnitude of the effect.

![WebUI Attention]({{ '/assets/images/37/webui-attention.jpg' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Figure N. Attention. Web UI provides features to increase and decrease attention on specific words in text prompt.* [ref1].

### Guide on Running Web UI
This is a small guide on how to run stable diffusion models using [Stable Diffusion web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) on Mac. It is based on this [tutorial](https://stable-diffusion-art.com/install-mac/).

Step 1: Install Homebrew and add brew to your path
```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
Add brew to your path
```
echo 'eval $(/opt/homebrew/bin/brew shellenv)' >> /Users/$USER/.zprofile
eval $(/opt/homebrew/bin/brew shellenv)
```

Step 2: Install required packages
```
brew install python@3.10 git wget
```

Step 3: Clone AUTOMATIC1111 webui repo
```
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui [location on your disk]
```

A stable-diffusion-webui folder should’ve been created wherever you specified the git clone destination to be.

Step 4: Run AUTOMATIC1111 webui
```
cd [destination dir]/stable-diffusion-webui
./webui.sh
```

After a while a new browser page hosted at http://127.0.0.1:7860/ should open.

### Astronaut Cat: Playing Around with Stable Diffusion Web UI’s Text to Image Functionalities

Our team decided to explore web UI’s text to image functionalities using the prompt “astronaut cat” (because why not :)). 

![WebUI-1]({{ '/assets/images/37/webui-1.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Figure N. First look at web UI’s txt2img functionality with the prompt “astronaut cat.”* 

The first thing to note from Figure __ is that web UI allows us to change the Stable Diffusion checkpoint that we wish to use. By default following the guide above, the checkpoint is [v1-5-pruned-emaonly.safetensors](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.safetensors). See section _____ to learn how to change checkpoints.

Next, the various tabs right below the Stable Diffusion checkpoint field show the various features that web UI provides, including img2img, embedding and hypernetwork training, and a place to allow users to upload their own PNGs. We will be focusing on the txt2img tab.

![WebUI-2]({{ '/assets/images/37/webui-2.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Figure N. Web UI allows the user to change the sampling method, schedule type, sampling steps, batch size, CGF scale, seed, and much more for image generation, all in a nice interface.* 

For text to image, web UI provides many easy-to-use interfaces to change parameters for image generation. For example, you can specify the sampling method or the schedule type that you want. You can also specify the number of sampling steps, image size, batch count and size, CFG scale, and seed. You can also upload [custom scripts](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Custom-Scripts) that can add even more functionality to web UI such as [generating color palettes](https://github.com/1ort/txt2palette) by text prompt or [specifying inpainting mask through text](https://github.com/ThereforeGames/txt2mask) rather than manually drawing it out.

![WebUI-3]({{ '/assets/images/37/webui-3.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Figure N. Astronaut cat with some modified parameters.* 

Playing around with some parameters (which is easy with the slider interface), we tested increasing the sampling steps by a little and increasing the CFG scale to 15, as well as setting the seed to 61. Figure ____ shows the results, which shows the cat in a forward-facing position and more realistic features (although the suit remains animated). 

![WebUI-4]({{ '/assets/images/37/webui-4.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Figure N. Making the image more realistic by increasing sampling steps and CFG scale, as well as changing the prompt to “realistic astronaut cat”.* 

Changing the prompt to “realistic astronaut cat”, increasing the sampling steps, and increasing the CFG scale even more produces an even more realistic cat, and this time the suit has some 3-dimensional shading and the background is more detailed.

![WebUI-5]({{ '/assets/images/37/webui-5.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Figure N. Attempt to produce a cute astronaut cat.* 

Finally, in the desire to produce a cute astronaut cat, we changed the prompt to “realistic cute astronaut cat” and set the seed to random, which produced this rather cute cat with an astronaut helmet with cat ears! 

## References  
[x] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L. Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, Jonathan Ho, David J. Fleet, and Mohammad Norouzi. ["Photorealistic text-to-image diffusion models with deep language understanding."](https://arxiv.org/abs/2205.11487) *arXiv preprint arXiv:2205.11487* (2022).
[ref]
[ref1]

---
