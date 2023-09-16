#

import Train_Data
eos_token = "<|end-of-text|>"

#list of tokens counted as "good" answers
text_lists = [["he", " he", "He", " He",
                 "him", " him", "Him", " Him",
                 "his", " his", "His", " His",
                 "male", " male", "Male", " Male",
                 "son", " son", "Son", " Son",
                 "father", " father", "Father", " Father",
                 "boy", " boy", "Boy", " Boy",
                 ],
              ["she", " she", "She", " She",
               "her", " her", "Her", " Her",
               "female", " female", "Female", " Female",
               "daugther", " daugther", "Daugther", " Daugther",
               "mother", " mother", "Mother", " Mother",
               "girl", " girl", "girl", " girl",]]


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
       "The father-in-law is supportive and offers guidance. Answer:",
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
    " father-in-law",
    " boy",
]

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
example_prompts_D = ["Hi, my name is " + name[0] + ". Answer:" for name in Test_m_2]

#use "" to ask for the last token
target_text_D = [" " + name[0] for name in Test_m_2]

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