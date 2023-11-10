#Code to generate de training data, run init_dataset.py to generate it.
import pandas as pd
from itertools import product
from Data.Test_Data import add_preprompt
eos_token = '<|endoftext|>'

def generate_trainset_v1(threshold=20):
  '''
  Creation of the dataset D_{noun, pronoun, anatomy}: it contains gendered nouns, pronouns, and anatomy words.
  There is a total of 240 distinct pairs of gendered words.
  Each type of gendered word is embedded into different prompts.

  Creation of the dataset D_name: it contains gendered names taken from the dataset "Multi-Dimensional Gender Bias Classification", year 1880.
  There is a total of ~750 distinct names for each gender, but names are not paired.
  They are combined with 3 prompts each.
  '''

  D_name = pd.read_csv("Data/gendered_names.csv")

  D_noun_m = ['countryman',   'wizards', 'manservant',  'fathers', 'divo', 'actor',   'bachelor', 'papa', 'dukes',     'barman',   'countrymen',   'brideprice', 'hosts',     'airmen',   'prince',   'governors',   'abbot',  'men',   'widower', 'gentlemen', 'sorcerers',   'bridegrooms', 'baron',    'househusbands', 'gods',      'nephew', 'widowers', 'lord', 'brother', 'grooms', 'priest', 'bellboys',  'marquis',     'princes',    'emperors',  'stallion', 'chairman',   'monastery', 'priests',     'boyhood',  'fellas', 'king',  'dudes', 'daddies', 'manservant', 'spokesman',    'tailor',     'cowboys',  'dude',   'bachelors', 'emperor', 'daddy', 'masculism', 'guys', 'enchanter',    'guy', 'fatherhood', 'androgen', 'cameramen',   'godfather', 'strongman',   'god',      'patriarch', 'uncle', 'chairmen',   'brotherhood',  'host',    'husband', 'dad', 'steward',    'males',   'spokesmen',   'pa', 'beau',  'stud', 'bachelor', 'wizard', 'sir',  'nephews',  'bull',     'beaus', 'councilmen',    'landlords',  'grandson',       'fiances',  'stepfathers', 'horsemen',    'grandfathers', 'schoolboy',  'rooster',  'grandsons',      'bachelor',     'cameraman',    'dads', 'master',   'lad',  'policeman',    'monk', 'actors',    'salesmen',    'boyfriend',  'councilman',   'fella',  'statesman',    'paternal', 'chap', 'landlord', 'brethren', 'lords',   'bellboy',   'duke',    'ballet dancer', 'dudes',  'fiance',   'colts',    'husbands', 'suitor',   'businessman',    'masseurs',   'hero',     'deer', 'busboys',  'boyfriends',   'kings',  'brothers', 'masters',    'stepfather', 'grooms', 'son',      'studs',  'cowboy',   'mentleman',  'sons',       'baritone', 'salesman',   'paramour', 'male_host',  'monks',  'menservants',  'headmasters',    'lads',   'congressman',    'airman',   'househusband', 'priest',     'barmen',   'barons',     'handyman',   'beard', 'stewards',      'colt',   'czar',     'stepsons',       'boys',   'lions',      'gentleman', 'masseur',  'bulls',  'uncles', 'bloke', 'beards', 'hubby', 'lion',     'sorcerer',  'father',  'males',    'waiters',    'stepson',      'businessmen',    'heir',     'waiter',   'headmaster',   'man',    'governor',   'god',      'bridegroom', 'grandpa', 'groom', 'dude', 'gents', 'boy',   'grandfather', 'gelding', 'paternity', 'roosters', 'priests', 'manservants',  'busboy',  'heros',    'fraternal', 'adultry',   'fraternity', 'fraternities', 'tailors',      'abbots']
  D_noun_f = ['countrywoman', 'witches', 'maidservant', 'mothers', 'diva', 'actress', 'spinster', 'mama', 'duchesses', 'barwoman', 'countrywomen', 'dowry',      'hostesses', 'airwomen', 'princess', 'governesses', 'abbess', 'women', 'widow',   'ladies',    'sorceresses', 'brides',      'baroness', 'housewives',    'goddesses', 'niece',  'widows',   'lady', 'sister',  'brides', 'nun',    'bellgirls', 'marchioness', 'princesses', 'empresses', 'mare',     'chairwoman', 'convent',   'priestesses', 'girlhood', 'ladies', 'queen', 'gals',  'mommies', 'maid',       'spokeswoman',  'seamstress', 'cowgirls', 'chick',  'spinsters', 'empress', 'mommy', 'feminism',  'gals', 'enchantress',  'gal', 'motherhood', 'estrogen', 'camerawomen', 'godmother', 'strongwoman', 'goddess',  'matriarch', 'aunt',  'chairwomen', 'sisterhood',   'hostess', 'wife',    'mom', 'stewardess', 'females', 'spokeswomen', 'ma', 'belle', 'minx', 'maiden',   'witch',  'miss', 'nieces',   'mothered', 'belles', 'councilwomen', 'landladies', 'granddaughter',  'fiancees', 'stepmothers', 'horsewomen',  'grandmothers', 'schoolgirl', 'hen',      'granddaughters', 'bachelorette', 'camerawoman',  'moms', 'mistress', 'lass', 'policewoman',  'nun',  'actresses', 'saleswomen',  'girlfriend', 'councilwoman', 'lady',   'stateswoman',  'maternal', 'lass', 'landlady', 'sistren',  'duchess', 'bellgirl',  'duchess', 'ballerina',     'chicks', 'fiancee',  'fillies',  'wives',    'suitress', 'businesswoman',  'masseuses',  'heroine',  'doe',  'busgirls', 'girlfriends',  'queens', 'sisters',  'mistresses', 'stepmother', 'brides', 'daughter', 'minxes', 'cowgirl',  'lady',       'daughters',  'mezzo',    'saleswoman', 'mistress', 'hostess',    'nuns',   'maids',        'headmistresses', 'lasses', 'congresswoman',  'airwoman', 'housewife',    'priestess',  'barwomen', 'baronesses', 'handywoman', 'toque', 'stewardesses',  'filly',  'czarina',  'stepdaughters',  'girls',  'lionesses',  'lady',      'masseuse', 'cows',   'aunts',  'wench', 'toques', 'wife',  'lioness',  'sorceress',  'mother', 'females',  'waitresses', 'stepdaughter', 'businesswomen',  'heiress',  'waitress', 'headmistress', 'woman',  'governess',  'goddess',  'bride',      'grandma', 'bride', 'gal', 'ladies', 'girl',  'grandmother', 'mare',    'maternity', 'hens',     'nuns',    'maidservants', 'busgirl', 'heroines', 'sororal',   'adultress', 'sorority',   'sororities',   'seamstresses', 'abbesses']

  D_anatomy_m = ['andropause', 'glans penis',  'testosterone', 'penis',  'sperm', 'prostate',    'urethra',]
  D_anatomy_f = ['menopause',  'clitoris',     'estradiol',    'vagina', 'ovum',  'skene gland', 'uterus',]

  D_pronoun_m = [' He', ' His', 'He', 'His', ' he', ' his', 'he', 'his']
  D_pronoun_f = [' She', ' Her', 'She', 'Her', ' she', ' her', 'she', 'her']

  D_name_f = D_name[D_name['assigned_gender'] == 1][D_name['count']>threshold]['name']
  D_name_m = D_name[D_name['assigned_gender'] == 0][D_name['count']>threshold]['name']


  prompt_noun = [
      'The ',
      ' ',
      'It doesn\'t matter who you are, you can be the ',
      'My friend often dresses like a ',
      'I like this movie because it features my favorite comedian playing the ',
      'I was scared by these people until realized it was just the ',
      'I usually don\'t like crowded places, but this time the ',
  ]

  prompt_pronoun = [
      'The detective examined the crime scene.',
      'The teacher stood at the front of the classroom.',
      'The chef prepared a delicious meal.',
      'The writer typed away on the keyboard.',
      'The artist applied vibrant colors to the canvas.',
      'The pilot expertly maneuvered the plane the sky.',
      'The scientist conducted experiments.',
      'The firefighter rushed into the building.',
      'The athlete trained rigorously.',
      'The electrician checked all the wiring.',
      'The gardener tended to the vegetables.',
      'The photographer captured stunning images.',
      'The mechanic fixed the engine.',
      'The surgeon performed the operation.',
      'The runner completed the marathon.',
      'The plumber unclogged the pipe.',
      'The veterinarian gently cared for the animal.',
      'The construction worker built the house.',
      'The cashier scanned all the items.',
      'The dancer performed an elegant routine.',
      'The musician played a beautiful melody.',
      'The architect designed an innovative building.',
      'The farmer harvested the ripe crops.',
      'The custodian cleaned the entire office building.',
      '',
  ]

  prompt_anatomy = [
      'The doctor told me I have a problem with my ',
      'The medical term is ',
      'The ',
      ' ',
  ]

  prompt_name_m = ['',
                  'My name is ',
                  'His name is ']
  prompt_name_f = ['',
                  'My name is ',
                  'Her name is ']


  D_pro_m, D_pro_f, len_pro_m, len_pro_f = add_preprompt(D_pronoun_m, D_pronoun_f, prompt_pronoun, prompt_pronoun)
  D_noun_m, D_noun_f, len_noun_m, len_noun_f = add_preprompt(D_noun_m, D_noun_f, prompt_noun, prompt_noun)
  D_anat_m, D_anat_f, len_anat_m, len_anat_f = add_preprompt(D_anatomy_m, D_anatomy_f, prompt_anatomy, prompt_anatomy)
  D_name_m, D_name_f, len_name_m, len_name_f = add_preprompt(D_name_m, D_name_f, prompt_name_m, prompt_name_f)


  examples = D_pro_m + D_pro_f + D_noun_m + D_noun_f + D_anat_m + D_anat_f + D_name_m + D_name_f
  data_lbl = ['pronouns']*(len_pro_m+len_pro_f) + ['nouns']*(len_noun_m+len_noun_f) + ['anatomy']*(len_anat_m+len_anat_f) + ['name']*(len_name_m+len_name_f)
  bin = [1]*len_pro_m + [-1]*len_pro_f + [1]*len_noun_m + [-1]*len_noun_f + [1]*len_anat_m + [-1]*len_anat_f + [1]*len_name_m + [-1]*len_name_f

  dataset = {
    'examples': examples,
    'label': data_lbl,
    'bin': bin,
  }

  dataframe = pd.DataFrame.from_dict(dataset, orient = 'columns')
  pd.DataFrame.to_csv(dataframe, "Data/Train_Data_v1.csv", quotechar='"')


def add_preprompt(D_m, D_f, prompts_m, prompts_f, end_m = "", end_f = ""):
    Data_m = [eos_token + prpt + word + end_m for word, prpt in product(D_m, prompts_m)]
    Data_f = [eos_token + prpt + word + end_f for word, prpt in product(D_f, prompts_f)]
    len_m, len_f = len(Data_m), len(Data_f)
    return Data_m, Data_f, len_m, len_f


def generate_trainset_v2(threshold=20):
  D_name = pd.read_csv("Data/gendered_names.csv")

  D_noun_M = ['countryman',   'wizard', 'manservant',  'father', 'divo', 'actor',   'bachelor', 'papa', 'duke',     'barman',   'brideprice', 'host',     'airman',   'prince',   'governor',   'abbot',  'man',   'widower', 'gentleman', 'sorcerer',   'bridegroom', 'baron',    'househusband', 'god',      'nephew', 'widower', 'lord', 'brother', 'groom', 'priest', 'bellboy',  'marquis',     'emperor',  'stallion', 'chairman',   'monastery', 'priest',     'boyhood',  'fella', 'king',  'dude', 'daddie', 'manservant', 'spokesman',    'tailor',     'cowboy',  'dude',   'bachelor', 'emperor', 'daddy', 'masculism', 'guys', 'enchanter',   'fatherhood', 'androgen', 'cameramen',   'godfather', 'strongman',    'patriarch', 'uncle', 'chairman',   'brotherhood', 'husband', 'dad', 'steward',    'male',   'spokesman',   'pa', 'beau',  'stud', 'bachelor', 'wizard', 'sir',  'nephew',  'bull',     'beau', 'councilman',    'landlord',  'grandson',       'fiance',  'stepfather', 'horseman',    'grandfather', 'schoolboy',  'rooster',  'grandson',      'bachelor',     'cameraman',    'dad', 'master',   'lad',  'policeman',    'monk', 'actors',    'salesman',    'boyfriend',  'councilman',   'fella',  'statesman',    'paternal', 'chap', 'landlord', 'brethren', 'lord',   'bellboy',   'duke',    'ballet dancer', 'dude',   'colts',   'suitor',   'businessman',    'masseur',   'hero',     'deer', 'busboy',  'boyfriend',   'king',  'brother', 'master',    'stepfather', 'son',      'stud',  'cowboy',   'mentleman',  'son',       'baritone', 'salesman',   'paramour', 'male_host',  'monk',  'menservant',  'headmaster',    'lad',   'congressman',    'airman',   'househusband', 'priest',     'barmen',   'baron',     'handyman',   'beard', 'steward',      'colt',   'czar',     'stepson',       'boy',   'lion',      'gentleman', 'bulls',  'uncle', 'bloke', 'beard', 'hubby', 'lion',     'sorcerer',  'father',  'male',    'waiter',    'stepson',      'businessman',    'heir',     'waiter',   'headmaster',   'man',    'governor',  'bridegroom', 'grandpa', 'groom', 'dude', 'gent', 'boy',   'grandfather', 'gelding', 'paternity', 'rooster', 'priest', 'manservant',  'busboy',  'hero',    'fraternal', 'adultry',   'fraternity', 'fraternities', 'tailor',    ]
  D_noun_F = ['countrywoman', 'witche', 'maidservant', 'mother', 'diva', 'actress', 'spinster', 'mama', 'duchesse', 'barwoman', 'dowry',      'hostesse', 'airwoman', 'princess', 'governesse', 'abbess', 'woman', 'widow',   'ladie',     'sorceresse', 'bride',      'baroness', 'housewife',    'goddesse', 'niece',  'widow',   'lady', 'sister',  'bride', 'nun',    'bellgirl', 'marchioness', 'empresse', 'mare',     'chairwoman', 'convent',   'priestesse', 'girlhood', 'ladie', 'queen', 'gal',  'mommie', 'maid',       'spokeswoman',  'seamstress', 'cowgirl', 'chick',  'spinster', 'empress', 'mommy', 'feminism',  'gals', 'enchantress',  'motherhood', 'estrogen', 'camerawomen', 'godmother', 'strongwoman', 'matriarch', 'aunt',  'chairwoman', 'sisterhood',  'wife',    'mom', 'stewardess', 'female', 'spokeswoman', 'ma', 'belle', 'minx', 'maiden',   'witch',  'miss', 'niece',   'mothered', 'belle', 'councilwoman', 'landladie', 'granddaughter',  'fiancee', 'stepmother', 'horsewoman',  'grandmother', 'schoolgirl', 'hen',      'granddaughter', 'bachelorette', 'camerawoman',  'mom', 'mistress', 'lass', 'policewoman',  'nun',  'actresses', 'saleswoman',  'girlfriend', 'councilwoman', 'lady',   'stateswoman',  'maternal', 'lass', 'landlady', 'sistren',  'duchess', 'bellgirl',  'duchess', 'ballerina',     'chick', 'fillies', 'suitress', 'businesswoman',  'masseuse',  'heroine',  'doe',  'busgirl', 'girlfriend',  'queen', 'sister',  'mistresse', 'stepmother', 'daughter', 'minxe', 'cowgirl',  'lady',       'daughter',  'mezzo',    'saleswoman', 'mistress', 'hostess',    'nun',   'maid',        'headmistresse', 'lasse', 'congresswoman',  'airwoman', 'housewife',    'priestess',  'barwomen', 'baronesse', 'handywoman', 'toque', 'stewardesse',  'filly',  'czarina',  'stepdaughter',  'girl',  'lionesse',  'lady',      'cows',   'aunt',  'wench', 'toque', 'wife',  'lioness',  'sorceress',  'mother', 'female',  'waitresse', 'stepdaughter', 'businesswoman',  'heiress',  'waitress', 'headmistress', 'woman',  'governess', 'bride',      'grandma', 'bride', 'gal', 'ladie', 'girl',  'grandmother', 'mare',    'maternity', 'hen',     'nun',    'maidservant', 'busgirl', 'heroine', 'sororal',   'adultress', 'sorority',   'sororities',   'seamstresse']

  D_anatomy_m = ['an andropause', 'a penis gland',  'testosterone', 'a penis',  'sperm', 'a prostate',    'a urethra',]
  D_anatomy_f = ['a menopause',  'a clitoris',     'estradiol',    'a vagina', 'ovum',  'a skene gland', 'a uterus',]

  D_pronoun_m = ['his', 'him']
  D_pronoun_f = ['her']

  D_name_F = D_name[D_name['assigned_gender'] == 1][D_name['count']>threshold]['name']
  D_name_M = D_name[D_name['assigned_gender'] == 0][D_name['count']>threshold]['name']


  prompt_ab = "The task is to predict the gender of the character in the following sentences. Answer only using A for \'female\' or B for \'male\'. \n "
  prompt_ba = "The task is to predict the gender of the character in the following sentences. Answer only using B for \'female\' or A for \'male\'. \n "

  prompt_a = "He is alive"
  prompt_b = "She is alive"

  answer_a = ". Answer: A"
  answer_b = ". Answer: B"

  prompt_ab = [prompt_ab + prompt_a + answer_b + " \n " + prompt_b + answer_a + " \n ",
               prompt_ab + prompt_b + answer_a + " \n " + prompt_a + answer_b + " \n "]
  prompt_ba = [prompt_ba + prompt_a + answer_a + " \n " + prompt_b + answer_b + " \n ",
               prompt_ba + prompt_b + answer_b + " \n " + prompt_a + answer_a + " \n "]
  

  prompt_noun = [prompt_ab[0] + "I am a ", prompt_ab[1] + "I am a "]
  prompt_pronoun = [prompt_ab[0] + "I like ", prompt_ab[1] + "I like "]
  prompt_anatomy = [prompt_ab[0] + "I have ", prompt_ab[1] + "I have "]
  prompt_name = [prompt_ab[0] + "My name is ", prompt_ab[1] + "My name is "]

  D_pro_m, D_pro_f, len_pro_m, len_pro_f = add_preprompt(D_pronoun_m, D_pronoun_f, prompt_pronoun, prompt_pronoun, end_m = answer_b, end_f = answer_a)
  D_noun_m, D_noun_f, len_noun_m, len_noun_f = add_preprompt(D_noun_M, D_noun_F, prompt_noun, prompt_noun, end_m = answer_b, end_f = answer_a)
  D_anat_m, D_anat_f, len_anat_m, len_anat_f = add_preprompt(D_anatomy_m, D_anatomy_f, prompt_anatomy, prompt_anatomy, end_m = answer_b, end_f = answer_a)
  D_name_m, D_name_f, len_name_m, len_name_f = add_preprompt(D_name_M, D_name_F, prompt_name, prompt_name, end_m = answer_b, end_f = answer_a)


  prompt_noun = [prompt_ba[0] + "I am a ", prompt_ba[1] + "I am a "]
  prompt_pronoun = [prompt_ba[0] + "I like ", prompt_ba[1] + "I like "]
  prompt_anatomy = [prompt_ba[0] + "I have ", prompt_ba[1] + "I have "]
  prompt_name = [prompt_ba[0] + "My name is ", prompt_ba[1] + "My name is "]

  a, b, c, d = add_preprompt(D_pronoun_m, D_pronoun_f, prompt_pronoun, prompt_pronoun, end_m = answer_a, end_f = answer_b)
  D_pro_m += a
  D_pro_f += b 
  len_pro_m += c 
  len_pro_f += d
  a, b, c, d = add_preprompt(D_noun_M, D_noun_F, prompt_noun, prompt_noun, end_m = answer_a, end_f = answer_b)
  D_noun_m += a
  D_noun_f += b 
  len_noun_m += c 
  len_noun_f += d
  a, b, c, d = add_preprompt(D_anatomy_m, D_anatomy_f, prompt_anatomy, prompt_anatomy, end_m = answer_a, end_f = answer_b)
  D_anat_m += a
  D_anat_f += b 
  len_anat_m += c 
  len_anat_f += d
  a, b, c, d = add_preprompt(D_name_M, D_name_F, prompt_name, prompt_name, end_m = answer_a, end_f = answer_b)
  D_name_m += a
  D_name_f += b 
  len_name_m += c 
  len_name_f += d


  examples = D_pro_m + D_pro_f + D_noun_m + D_noun_f + D_anat_m + D_anat_f + D_name_m + D_name_f
  data_lbl = ['pronouns']*(len_pro_m+len_pro_f) + ['nouns']*(len_noun_m+len_noun_f) + ['anatomy']*(len_anat_m+len_anat_f) + ['name']*(len_name_m+len_name_f)
  bin = [1]*len_pro_m + [-1]*len_pro_f + [1]*len_noun_m + [-1]*len_noun_f + [1]*len_anat_m + [-1]*len_anat_f + [1]*len_name_m + [-1]*len_name_f

  dataset = {
    'examples': examples,
    'label': data_lbl,
    'bin': bin,
  }

  dataframe = pd.DataFrame.from_dict(dataset, orient = 'columns')
  pd.DataFrame.to_csv(dataframe, "Data/Train_Data_v2.csv", quotechar='"')

  print("Number of example ", len(examples))



def generate_dataset_v3(threshold=50):
  D_name = pd.read_csv("Data/gendered_names.csv")

  D_noun_m = ['countryman',   'wizards', 'manservant',  'fathers', 'divo', 'actor',   'bachelor', 'papa', 'dukes',     'barman',   'countrymen',   'brideprice', 'hosts',     'airmen',   'prince',   'governors',   'abbot',  'men',   'widower', 'gentlemen', 'sorcerers',   'bridegrooms', 'baron',    'househusbands', 'gods',      'nephew', 'widowers', 'lord', 'brother', 'grooms', 'priest', 'bellboys',  'marquis',     'princes',    'emperors',  'stallion', 'chairman',   'monastery', 'priests',     'boyhood',  'fellas', 'king',  'dudes', 'daddies', 'manservant', 'spokesman',    'tailor',     'cowboys',  'dude',   'bachelors', 'emperor', 'daddy', 'masculism', 'guys', 'enchanter',    'guy', 'fatherhood', 'androgen', 'cameramen',   'godfather', 'strongman',   'god',      'patriarch', 'uncle', 'chairmen',   'brotherhood',  'host',    'husband', 'dad', 'steward',    'males',   'spokesmen',   'pa', 'beau',  'stud', 'bachelor', 'wizard', 'sir',  'nephews',  'bull',     'beaus', 'councilmen',    'landlords',  'grandson',       'fiances',  'stepfathers', 'horsemen',    'grandfathers', 'schoolboy',  'rooster',  'grandsons',      'bachelor',     'cameraman',    'dads', 'master',   'lad',  'policeman',    'monk', 'actors',    'salesmen',    'boyfriend',  'councilman',   'fella',  'statesman',    'paternal', 'chap', 'landlord', 'brethren', 'lords',   'bellboy',   'duke',    'ballet dancer', 'dudes',  'fiance',   'colts',    'husbands', 'suitor',   'businessman',    'masseurs',   'hero',     'deer', 'busboys',  'boyfriends',   'kings',  'brothers', 'masters',    'stepfather', 'grooms', 'son',      'studs',  'cowboy',   'mentleman',  'sons',       'baritone', 'salesman',   'paramour', 'male_host',  'monks',  'menservants',  'headmasters',    'lads',   'congressman',    'airman',   'househusband', 'priest',     'barmen',   'barons',     'handyman',   'beard', 'stewards',      'colt',   'czar',     'stepsons',       'boys',   'lions',      'gentleman', 'masseur',  'bulls',  'uncles', 'bloke', 'beards', 'hubby', 'lion',     'sorcerer',  'father',  'males',    'waiters',    'stepson',      'businessmen',    'heir',     'waiter',   'headmaster',   'man',    'governor',   'god',      'bridegroom', 'grandpa', 'groom', 'dude', 'gents', 'boy',   'grandfather', 'gelding', 'paternity', 'roosters', 'priests', 'manservants',  'busboy',  'heros',    'fraternal', 'adultry',   'fraternity', 'fraternities', 'tailors',      'abbots']
  D_noun_f = ['countrywoman', 'witches', 'maidservant', 'mothers', 'diva', 'actress', 'spinster', 'mama', 'duchesses', 'barwoman', 'countrywomen', 'dowry',      'hostesses', 'airwomen', 'princess', 'governesses', 'abbess', 'women', 'widow',   'ladies',    'sorceresses', 'brides',      'baroness', 'housewives',    'goddesses', 'niece',  'widows',   'lady', 'sister',  'brides', 'nun',    'bellgirls', 'marchioness', 'princesses', 'empresses', 'mare',     'chairwoman', 'convent',   'priestesses', 'girlhood', 'ladies', 'queen', 'gals',  'mommies', 'maid',       'spokeswoman',  'seamstress', 'cowgirls', 'chick',  'spinsters', 'empress', 'mommy', 'feminism',  'gals', 'enchantress',  'gal', 'motherhood', 'estrogen', 'camerawomen', 'godmother', 'strongwoman', 'goddess',  'matriarch', 'aunt',  'chairwomen', 'sisterhood',   'hostess', 'wife',    'mom', 'stewardess', 'females', 'spokeswomen', 'ma', 'belle', 'minx', 'maiden',   'witch',  'miss', 'nieces',   'mothered', 'belles', 'councilwomen', 'landladies', 'granddaughter',  'fiancees', 'stepmothers', 'horsewomen',  'grandmothers', 'schoolgirl', 'hen',      'granddaughters', 'bachelorette', 'camerawoman',  'moms', 'mistress', 'lass', 'policewoman',  'nun',  'actresses', 'saleswomen',  'girlfriend', 'councilwoman', 'lady',   'stateswoman',  'maternal', 'lass', 'landlady', 'sistren',  'duchess', 'bellgirl',  'duchess', 'ballerina',     'chicks', 'fiancee',  'fillies',  'wives',    'suitress', 'businesswoman',  'masseuses',  'heroine',  'doe',  'busgirls', 'girlfriends',  'queens', 'sisters',  'mistresses', 'stepmother', 'brides', 'daughter', 'minxes', 'cowgirl',  'lady',       'daughters',  'mezzo',    'saleswoman', 'mistress', 'hostess',    'nuns',   'maids',        'headmistresses', 'lasses', 'congresswoman',  'airwoman', 'housewife',    'priestess',  'barwomen', 'baronesses', 'handywoman', 'toque', 'stewardesses',  'filly',  'czarina',  'stepdaughters',  'girls',  'lionesses',  'lady',      'masseuse', 'cows',   'aunts',  'wench', 'toques', 'wife',  'lioness',  'sorceress',  'mother', 'females',  'waitresses', 'stepdaughter', 'businesswomen',  'heiress',  'waitress', 'headmistress', 'woman',  'governess',  'goddess',  'bride',      'grandma', 'bride', 'gal', 'ladies', 'girl',  'grandmother', 'mare',    'maternity', 'hens',     'nuns',    'maidservants', 'busgirl', 'heroines', 'sororal',   'adultress', 'sorority',   'sororities',   'seamstresses', 'abbesses']

  D_pronoun_m = [' He', ' His', 'He', 'His', ' he', ' his', 'he', 'his']
  D_pronoun_f = [' She', ' Her', 'She', 'Her', ' she', ' her', 'she', 'her']
  
  D_name_f = D_name[D_name['assigned_gender'] == 1][D_name['count']>threshold]['name']
  D_name_m = D_name[D_name['assigned_gender'] == 0][D_name['count']>threshold]['name']

  prompt_noun = [
      'The ',
      ' ',
  ]

  prompt_pronoun = [
     '',
  ]

  prompt_name_m = [' ',
                  'My name is ',
                  'His name is ']
  prompt_name_f = [' ',
                  'My name is ',
                  'Her name is ']


  D_pro_m, D_pro_f, len_pro_m, len_pro_f = add_preprompt(D_pronoun_m, D_pronoun_f, prompt_pronoun, prompt_pronoun)
  D_noun_m, D_noun_f, len_noun_m, len_noun_f = add_preprompt(D_noun_m, D_noun_f, prompt_noun, prompt_noun)
  D_name_m, D_name_f, len_name_m, len_name_f = add_preprompt(D_name_m, D_name_f, prompt_name_m, prompt_name_f)


  examples = D_pro_m + D_pro_f + D_noun_m + D_noun_f + D_name_m + D_name_f
  data_lbl = ['pronouns']*(len_pro_m+len_pro_f) + ['nouns']*(len_noun_m+len_noun_f) + ['name']*(len_name_m+len_name_f)
  bin = [1]*len_pro_m + [-1]*len_pro_f + [1]*len_noun_m + [-1]*len_noun_f + [1]*len_name_m + [-1]*len_name_f

  dataset = {
    'examples': examples,
    'label': data_lbl,
    'bin': bin,
  }

  dataframe = pd.DataFrame.from_dict(dataset, orient = 'columns')
  pd.DataFrame.to_csv(dataframe, "Data/Train_Data_v3.csv", quotechar='"')