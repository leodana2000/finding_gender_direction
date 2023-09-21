#code to generate de test data

import pandas as pd
from finding_gender_direction import utils

D2 = pd.read_csv("finding_gender_direction/yob1880.csv")
threshold = 20
D2_f = D2[D2['assigned_gender'] == 1][D2['count']>threshold]['name']
D2_m = D2[D2['assigned_gender'] == 0][D2['count']>threshold]['name']

Train_f, Test_f = utils.split_list([word for word in D2_f])
Train_m, Test_m = utils.split_list([word for word in D2_m])

eos_token = "<|end-of-text|>"


#female sentences

#evaluation prompt
example_prompts_A = ["The young lady is talented and is a musician. Answer:",
       "The actress is skilled and performed flawlessly. Answer:",
       "The daughter is intelligent and loves to read. Answer:",
       "The bride is stunning and chose a beautiful gown. Answer:",
       "The princess is kind-hearted and helps others. Answer:",
       "The girl is diligent and scored top grades. Answer:",
       "The mother is caring and always puts family first. Answer:",
       "The princess is determined and trains tirelessly. Answer:",
       "Their mother is creative and invents unique recipes. Answer:",
       "The grandmother is wise and shares valuable stories. Answer:",
       "The lady is fashionable and has a great sense of style. Answer:",
       "The queen is powerful and rules with authority. Answer:",
       "The aunt is nurturing and takes care of the children. Answer:",
       "The heroine is brave and faces danger without fear. Answer:",
       "The woman is confident and speaks loud. Answer:",
       "The female CEO is successful and leads the company to prosperity. Answer:",
       "The girl is imaginative and creates fantastic stories. Answer:",
       ]


#use "" to ask for the last token
target_text_A = [
    " lady",
    " actress",
    " daughter",
    " bride",
    " princess",
    " girl",
    " mother",
    " princess",
    " mother",
    " grandmother",
    " lady",
    " queen",
    " aunt",
    " heroine",
    " woman",
    " female",
    " girl",
]
target_text_A += target_text_A

pre_prompt_a = "He is a boy. Answer: male \n "
pre_prompt_b = "She is a girl. Answer female \n "

pre_prompt = "The task is to predict the gender of the character in the following sentences. Answer only using the word 'female' or 'male'. \n "

example_prompts_A = [eos_token
                     + pre_prompt
                     + pre_prompt_b
                     + pre_prompt_a
                     + example for example in example_prompts_A] + [eos_token
                     + pre_prompt
                     + pre_prompt_a
                     + pre_prompt_b
                     + example for example in example_prompts_A]



#male sentences

#evaluation prompt
example_prompts_B = ["The young lord is talented. Answer:",
       "The actor is skilled. Answer:",
       "The son is intelligent. Answer:",
       "The prince is kind-hearted. Answer:",
       "The boy is diligent. Answer:",
       "The father is caring. Answer:",
       "The prince is determined. Answer:",
       "Their father is creative. Answer:",
       "The grandfather is wise. Answer:",
       "The king is powerful. Answer:",
       "The hero is brave and has no fear. Answer:",
       "The man is confident and speaks loud. Answer:",
       "The male CEO is successful and leads the company to prosperity. Answer:",
       "The boy is imaginative and creates fantastic stories. Answer:",
       ]


#use "" to ask for the last token
target_text_B = [
    " lord",
    " actor",
    " son",
    " prince",
    " boy",
    " father",
    " prince",
    " father",
    " grandfather",
    " king",
    " hero",
    " man",
    " male",
    " boy",
]
target_text_B += target_text_B

pre_prompt_a = "He is a boy. Answer: male \n "
pre_prompt_b = "She is a girl. Answer female \n "

pre_prompt = "The task is to predict the gender of the caracter in the following sentences. Answer only using the word 'female' or 'male'.\n"

example_prompts_B = [eos_token
                     + pre_prompt
                     + pre_prompt_b
                     + pre_prompt_a
                     + example for example in example_prompts_B] + [eos_token
                     + pre_prompt
                     + pre_prompt_a
                     + pre_prompt_b
                     + example for example in example_prompts_B]



#female names

#evaluation prompt: checked that all prompt are understood by GPT2-xl
example_prompts_C = ["Hi, my name is " + name + ". Answer:" for name in Test_f]

#use "" to ask for the last token
target_text_C = [" " + name for name in Test_f]
target_text_C += target_text_C

pre_prompt_a = "He is a boy. Answer: male \n "
pre_prompt_b = "She is a girl. Answer female \n "

pre_prompt = "The task is to predict the gender of the caracter in the following sentences. Answer only using the word 'female' or 'male'.\n"

example_prompts_C = [eos_token
                     + pre_prompt
                     + pre_prompt_b
                     + pre_prompt_a
                     + example for example in example_prompts_C] + [eos_token
                     + pre_prompt
                     + pre_prompt_a
                     + pre_prompt_b
                     + example for example in example_prompts_C]




#male names
#!! doesn't recognize correctly all of them !!

#evaluation prompt: checked that all prompt are understood by GPT2-xl
example_prompts_D = ["Hi, my name is " + name[0] + ". Answer:" for name in Test_m]

#use "" to ask for the last token
target_text_D = [" " + name[0] for name in Test_m]
target_text_D += target_text_D

pre_prompt_a = "He is a boy. Answer: male \n "
pre_prompt_b = "She is a girl. Answer female \n "

pre_prompt = "The task is to predict the gender of the caracter in the following sentences. Answer only using the word 'female' or 'male'.\n"

example_prompts_D = [eos_token
                     + pre_prompt
                     + pre_prompt_b
                     + pre_prompt_a
                     + example for example in example_prompts_D] + [eos_token
                     + pre_prompt
                     + pre_prompt_a
                     + pre_prompt_b
                     + example for example in example_prompts_D]

D = {
    'question': example_prompts_A + example_prompts_B + example_prompts_C + example_prompts_D,
    'data_num': ['a']*len(example_prompts_A) + ['b']*len(example_prompts_B) + ['c']*len(example_prompts_C) + ['d']*len(example_prompts_D),
    'target': target_text_A + target_text_B + target_text_C + target_text_D,
}
DF = pd.DataFrame.from_dict(D, orient = 'columns')
pd.DataFrame.to_csv(DF, "finding_gender_direction/Test_Data.csv")
