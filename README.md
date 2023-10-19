# finding-gender-direction
This repository contains my code to find the gender direction in a language model like gpt2-xl. This is still a work in progress You can use it directly on google-colab by running the exp_notebook.ipynb.

This project is my internship work at SERI MATS 4.0 and FAR AI, in summer 2023. The goal was to find the gender direction on gpt2-xl and show if it is possible to remove gender in different ways using a method similar to [Inference Time Intervention](https://www.researchgate.net/publication/371347185_Inference-Time_Intervention_Eliciting_Truthful_Answers_from_a_Language_Model).\
More precisely, I learn a d-dimensional plane and a diagonal projection on that plane with the [LEACE](https://arxiv.org/abs/2306.03819), which I can then use to do activation steering.

You can find more on the techniques and findings in Intership_Report_Leo_Dana.pdf.