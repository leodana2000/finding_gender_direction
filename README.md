# finding-gender-direction
This repository contains my code to find the gender direction in a language model like gpt2-xl. This is still a work in progress You can use it directly on gg-colab with exp_notebook.ipynb.

This project is my internship work at SERI MATS 4.0 and FAR AI, in summer 2023. The goal was to find the gender direction on gpt2-xl and show if it is possible to remove gender in different ways using a method similar to Inference Time Intervention.
I thus test several technics available here:
* Classification of the activation vectors
* Classification of the pairwise activation vectors
* Intervention on one layer in the residual stream
* Intervention at all layer in the attention mecanism
* Intervention in a text generation setting

Explanation and graphs can be found here XYZ.