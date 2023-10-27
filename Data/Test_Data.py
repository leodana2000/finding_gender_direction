#Code to generate de test data, run init_dataset.py to generate it.
import pandas as pd
import hyperplane_computation.utils as utils

'''
Threshold is used so the names are 'real names'. 
A lower threshold means weirder names, which may not be recognize as the belonging to the correct gender.
'''

D2 = pd.read_csv('Data/gendered_names.csv')
threshold = 20 
Test_f = D2[D2['assigned_gender'] == 1][D2['count']>threshold]['name'].values.tolist()
Test_m = D2[D2['assigned_gender'] == 0][D2['count']>threshold]['name'].values.tolist()
eos_token = '<|endoftext|>'


'''
Creation of the datasets.
A: female nouns,
B: male nouns,
C: female names,
D: male names,

For each dataset, we have a list of sentences that should be answered by a gendered word.
Each sentence has one targeted word on which we use the intervention.
We also have a pre-prompt and few shot example to make the model understand its task.
Each sentences is used with both ordering of the few shots to avoid biases.
All sentences with pre-prompt and few-shots were tested and have a higher probability of True answer.
'''

example_prompts_A = ['The young lady is talented and is a musician. Answer:',
       'The actress is skilled and performed flawlessly. Answer:',
       'The daughter is intelligent and loves to read. Answer:',
       'The bride is stunning and chose a beautiful gown. Answer:',
       'The princess is kind-hearted and helps others. Answer:',
       'The girl is diligent and scored top grades. Answer:',
       'The mother is caring and always puts family first. Answer:',
       'The princess is determined and trains tirelessly. Answer:',
       'Their mother is creative and invents unique recipes. Answer:',
       'The grandmother is wise and shares valuable stories. Answer:',
       'The lady is fashionable and has a great sense of style. Answer:',
       'The queen is powerful and rules with authority. Answer:',
       'The aunt is nurturing and takes care of the children. Answer:',
       'The heroine is brave and faces danger without fear. Answer:',
       'The woman is confident and speaks loud. Answer:',
       'The female CEO is successful and leads the company to prosperity. Answer:',
       'The girl is imaginative and creates fantastic stories. Answer:',
       ]

target_text_A = [
    ' lady',
    ' actress',
    ' daughter',
    ' bride',
    ' princess',
    ' girl',
    ' mother',
    ' princess',
    ' mother',
    ' grandmother',
    ' lady',
    ' queen',
    ' aunt',
    ' heroine',
    ' woman',
    ' female',
    ' girl',
]
target_text_A += target_text_A

pre_prompt_a = 'He is a boy. Answer: male \n '
pre_prompt_b = 'She is a girl. Answer: female \n '

pre_prompt = 'The task is to predict the gender of the character in the following sentences. Answer only using the word \'female\' or \'male\'. \n '

example_prompts_A = [eos_token
                     + pre_prompt
                     + pre_prompt_b
                     + pre_prompt_a
                     + example for example in example_prompts_A] + [eos_token
                     + pre_prompt
                     + pre_prompt_a
                     + pre_prompt_b
                     + example for example in example_prompts_A]



example_prompts_B = [
        'The young lord is talented. Answer:',
        'The actor is skilled. Answer:',
        'The son is intelligent. Answer:',
        'The prince is kind-hearted. Answer:',
        'The boy is diligent. Answer:',
        'The father is caring. Answer:',
        'The prince is determined. Answer:',
        'Their father is creative. Answer:',
        'The grandfather is wise. Answer:',
        'The king is powerful. Answer:',
        'The hero is brave and has no fear. Answer:',
        'The man is confident and speaks loud. Answer:',
        'The male CEO is successful and leads the company to prosperity. Answer:',
        'The boy is imaginative and creates fantastic stories. Answer:',
        ]

target_text_B = [
    ' lord',
    ' actor',
    ' son',
    ' prince',
    ' boy',
    ' father',
    ' prince',
    ' father',
    ' grandfather',
    ' king',
    ' hero',
    ' man',
    ' male',
    ' boy',
]
target_text_B += target_text_B

pre_prompt_a = 'He is a boy. Answer: male \n '
pre_prompt_b = 'She is a girl. Answer: female \n '

pre_prompt = 'The task is to predict the gender of the character in the following sentences. Answer only using the word \'female\' or \'male\'. \n '

example_prompts_B = [eos_token
                     + pre_prompt
                     + pre_prompt_b
                     + pre_prompt_a
                     + example for example in example_prompts_B] + [eos_token
                     + pre_prompt
                     + pre_prompt_a
                     + pre_prompt_b
                     + example for example in example_prompts_B]



example_prompts_C = ['Hi, my name is ' + name + '. Answer:' for name in Test_f]

target_text_C = [" " + name for name in Test_f]
target_text_C += target_text_C

pre_prompt_a = 'He is a boy. Answer: male \n '
pre_prompt_b = 'She is a girl. Answer: female \n '

pre_prompt = 'The task is to predict the gender of the character in the following sentences. Answer only using the word \'female\' or \'male\'. \n '

example_prompts_C = [eos_token
                     + pre_prompt
                     + pre_prompt_b
                     + pre_prompt_a
                     + example for example in example_prompts_C] + [eos_token
                     + pre_prompt
                     + pre_prompt_a
                     + pre_prompt_b
                     + example for example in example_prompts_C]



example_prompts_D = ['Hi, my name is ' + name + '. Answer:' for name in Test_m]

target_text_D = [' ' + name for name in Test_m]
target_text_D += target_text_D

pre_prompt_a = 'He is a boy. Answer: male \n '
pre_prompt_b = 'She is a girl. Answer: female \n '

pre_prompt = 'The task is to predict the gender of the character in the following sentences. Answer only using the word \'female\' or \'male\'. \n '

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
    'bin': [-1]*len(example_prompts_A) + [1]*len(example_prompts_B) + [-1]*len(example_prompts_C) + [1]*len(example_prompts_D)
}
DF = pd.DataFrame.from_dict(D, orient = 'columns')
pd.DataFrame.to_csv(DF, "Data/Test_Data.csv")
