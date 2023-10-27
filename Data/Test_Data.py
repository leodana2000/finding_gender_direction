#Code to generate de test data, run init_dataset.py to generate it.
import pandas as pd
eos_token = '<|endoftext|>'

def generate_testset(threshold=20):

    '''
    Threshold is used so the names are 'real names'. 
    A lower threshold means weirder names, which may not be recognize as the belonging to the correct gender.
    '''

    D_name = pd.read_csv('Data/gendered_names.csv')
    threshold = threshold 
    D_name_f = D_name[D_name['assigned_gender'] == 1][D_name['count']>threshold]['name'].values.tolist()
    D_name_m = D_name[D_name['assigned_gender'] == 0][D_name['count']>threshold]['name'].values.tolist()


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
    example_prompts_A = add_preprompt(example_prompts_A)
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
    ]*2


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
    example_prompts_B = add_preprompt(example_prompts_B)
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
    ]*2


    example_prompts_C = add_preprompt(['Hi, my name is ' + name + '. Answer:' for name in D_name_f])
    target_text_C = [" " + name for name in D_name_f]*2


    example_prompts_D = add_preprompt(['Hi, my name is ' + name + '. Answer:' for name in D_name_m])
    target_text_D = [' ' + name for name in D_name_m]*2



    dataset = {
        'question': example_prompts_A + example_prompts_B + example_prompts_C + example_prompts_D,
        'data_num': ['a']*len(example_prompts_A) + ['b']*len(example_prompts_B) + ['c']*len(example_prompts_C) + ['d']*len(example_prompts_D),
        'target': target_text_A + target_text_B + target_text_C + target_text_D,
        'bin': [-1]*len(example_prompts_A) + [1]*len(example_prompts_B) + [-1]*len(example_prompts_C) + [1]*len(example_prompts_D)
    }
    dataframe = pd.DataFrame.from_dict(dataset, orient = 'columns')
    pd.DataFrame.to_csv(dataframe, "Data/Test_Data.csv")


def add_preprompt(example_prompts):
    """
    Transform the sentences into prompts, and return both few-shot learning order.
    """

    pre_prompt_a = 'He is a boy. Answer: male \n '
    pre_prompt_b = 'She is a girl. Answer: female \n '
    pre_prompt = 'The task is to predict the gender of the character in the following sentences. Answer only using the word \'female\' or \'male\'. \n '

    ab_prompts = [
        eos_token + pre_prompt + pre_prompt_a + pre_prompt_b + example for example in example_prompts
    ]
    ba_prompts = [
        eos_token + pre_prompt + pre_prompt_b + pre_prompt_a + example for example in example_prompts
    ]
    return ab_prompts + ba_prompts