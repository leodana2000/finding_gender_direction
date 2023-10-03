#code to generate de training data
import utils
import pandas as pd
eos_token = '<|endoftext|>'


#Creation of the dataset D1: it contains gendered words (nouns, pronouns, and anatomy).
#There is a total of 240 distinct pairs of gendered words.
#They are embedded into different prompts. 
#Executing the file gives the amount of final example for each classes.

D1_noun_m = ['countryman',   'wizards', 'manservant',  'fathers', 'divo', 'actor',   'bachelor', 'papa', 'dukes',     'barman',   'countrymen',   'brideprice', 'hosts',     'airmen',   'prince',   'governors',   'abbot',  'men',   'widower', 'gentlemen', 'sorcerers',   'bridegrooms', 'baron',    'househusbands', 'gods',      'nephew', 'widowers', 'lord', 'brother', 'grooms', 'priest', 'bellboys',  'marquis',     'princes',    'emperors',  'stallion', 'chairman',   'monastery', 'priests',     'boyhood',  'fellas', 'king',  'dudes', 'daddies', 'manservant', 'spokesman',    'tailor',     'cowboys',  'dude',   'bachelors', 'emperor', 'daddy', 'masculism', 'guys', 'enchanter',    'guy', 'fatherhood', 'androgen', 'cameramen',   'godfather', 'strongman',   'god',      'patriarch', 'uncle', 'chairmen',   'brotherhood',  'host',    'husband', 'dad', 'steward',    'males',   'spokesmen',   'pa', 'beau',  'stud', 'bachelor', 'wizard', 'sir',  'nephews',  'bull',     'beaus', 'councilmen',    'landlords',  'grandson',       'fiances',  'stepfathers', 'horsemen',    'grandfathers', 'schoolboy',  'rooster',  'grandsons',      'bachelor',     'cameraman',    'dads', 'master',   'lad',  'policeman',    'monk', 'actors',    'salesmen',    'boyfriend',  'councilman',   'fella',  'statesman',    'paternal', 'chap', 'landlord', 'brethren', 'lords',   'bellboy',   'duke',    'ballet dancer', 'dudes',  'fiance',   'colts',    'husbands', 'suitor',   'businessman',    'masseurs',   'hero',     'deer', 'busboys',  'boyfriends',   'kings',  'brothers', 'masters',    'stepfather', 'grooms', 'son',      'studs',  'cowboy',   'mentleman',  'sons',       'baritone', 'salesman',   'paramour', 'male_host',  'monks',  'menservants',  'headmasters',    'lads',   'congressman',    'airman',   'househusband', 'priest',     'barmen',   'barons',     'handyman',   'beard', 'stewards',      'colt',   'czar',     'stepsons',       'boys',   'lions',      'gentleman', 'masseur',  'bulls',  'uncles', 'bloke', 'beards', 'hubby', 'lion',     'sorcerer',  'father',  'males',    'waiters',    'stepson',      'businessmen',    'heir',     'waiter',   'headmaster',   'man',    'governor',   'god',      'bridegroom', 'grandpa', 'groom', 'dude', 'gents', 'boy',   'grandfather', 'gelding', 'paternity', 'roosters', 'priests', 'manservants',  'busboy',  'heros',    'fraternal', 'adultry',   'fraternity', 'fraternities', 'tailors',      'abbots']
D1_noun_f = ['countrywoman', 'witches', 'maidservant', 'mothers', 'diva', 'actress', 'spinster', 'mama', 'duchesses', 'barwoman', 'countrywomen', 'dowry',      'hostesses', 'airwomen', 'princess', 'governesses', 'abbess', 'women', 'widow',   'ladies',    'sorceresses', 'brides',      'baroness', 'housewives',    'goddesses', 'niece',  'widows',   'lady', 'sister',  'brides', 'nun',    'bellgirls', 'marchioness', 'princesses', 'empresses', 'mare',     'chairwoman', 'convent',   'priestesses', 'girlhood', 'ladies', 'queen', 'gals',  'mommies', 'maid',       'spokeswoman',  'seamstress', 'cowgirls', 'chick',  'spinsters', 'empress', 'mommy', 'feminism',  'gals', 'enchantress',  'gal', 'motherhood', 'estrogen', 'camerawomen', 'godmother', 'strongwoman', 'goddess',  'matriarch', 'aunt',  'chairwomen', 'sisterhood',   'hostess', 'wife',    'mom', 'stewardess', 'females', 'spokeswomen', 'ma', 'belle', 'minx', 'maiden',   'witch',  'miss', 'nieces',   'mothered', 'belles', 'councilwomen', 'landladies', 'granddaughter',  'fiancees', 'stepmothers', 'horsewomen',  'grandmothers', 'schoolgirl', 'hen',      'granddaughters', 'bachelorette', 'camerawoman',  'moms', 'mistress', 'lass', 'policewoman',  'nun',  'actresses', 'saleswomen',  'girlfriend', 'councilwoman', 'lady',   'stateswoman',  'maternal', 'lass', 'landlady', 'sistren',  'duchess', 'bellgirl',  'duchess', 'ballerina',     'chicks', 'fiancee',  'fillies',  'wives',    'suitress', 'businesswoman',  'masseuses',  'heroine',  'doe',  'busgirls', 'girlfriends',  'queens', 'sisters',  'mistresses', 'stepmother', 'brides', 'daughter', 'minxes', 'cowgirl',  'lady',       'daughters',  'mezzo',    'saleswoman', 'mistress', 'hostess',    'nuns',   'maids',        'headmistresses', 'lasses', 'congresswoman',  'airwoman', 'housewife',    'priestess',  'barwomen', 'baronesses', 'handywoman', 'toque', 'stewardesses',  'filly',  'czarina',  'stepdaughters',  'girls',  'lionesses',  'lady',      'masseuse', 'cows',   'aunts',  'wench', 'toques', 'wife',  'lioness',  'sorceress',  'mother', 'females',  'waitresses', 'stepdaughter', 'businesswomen',  'heiress',  'waitress', 'headmistress', 'woman',  'governess',  'goddess',  'bride',      'grandma', 'bride', 'gal', 'ladies', 'girl',  'grandmother', 'mare',    'maternity', 'hens',     'nuns',    'maidservants', 'busgirl', 'heroines', 'sororal',   'adultress', 'sorority',   'sororities',   'seamstresses', 'abbesses']

D1_anatomy_m = ['andropause', 'glans penis',  'testosterone', 'penis',  'sperm', 'prostate',    'urethra',]
D1_anatomy_f = ['menopause',  'clitoris',     'estradiol',    'vagina', 'ovum',  'skene gland', 'uterus',]

D1_pronoun_m = [' He', ' His', 'He', 'His', ' he', ' his', 'he', 'his']
D1_pronoun_f = [' She', ' Her', 'She', 'Her', ' she', ' her', 'she', 'her']

D1_m = [D1_pronoun_m, D1_noun_m, D1_anatomy_m]
D1_f = [D1_pronoun_f, D1_noun_f, D1_anatomy_f]


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

prompts = [prompt_pronoun, prompt_noun, prompt_anatomy]


D1_name = ['pronouns', 'nouns', 'anatomy']


Train_m_1 = []
Train_f_1 = []
Test_m_1 = []
Test_f_1 = []

for d1_m, d1_f, d1_name, prompt in zip(D1_m, D1_f, D1_name, prompts):
  Train_m, Test_m = utils.split_list(d1_m)
  Train_f, Test_f = utils.split_list(d1_f)

  Train_m = utils.concat_list([[eos_token + prpt + word for word in Train_m] for prpt in prompt])
  Train_f = utils.concat_list([[eos_token + prpt + word for word in Train_f] for prpt in prompt])
  Test_m = utils.concat_list([[eos_token + prpt + word for word in Test_m] for prpt in prompt])
  Test_f = utils.concat_list([[eos_token + prpt + word for word in Test_f] for prpt in prompt])

  Train_m_1 += Train_m
  Train_f_1 += Train_f
  Test_m_1 += Test_m
  Test_f_1 += Test_f

print(Train_m_1)
#Creation of the dataset D2: it contains gendered names.
#There is a total of ~750 distinct names.
#They are embedded into different prompts.
#Executing the file gives the amount of final example for each classes.

D2 = pd.read_csv("finding_gender_direction/yob1880.csv")
threshold = 20
D2_f = D2[D2['assigned_gender'] == 1][D2['count']>threshold]['name']
D2_m = D2[D2['assigned_gender'] == 0][D2['count']>threshold]['name']

prompt_m = ['',
            'My name is ',
            'His name is ']
prompt_f = ['',
            'My name is ',
            'Her name is ']

Train_f, Test_f = utils.split_list([word for word in D2_f])
Train_m, Test_m = utils.split_list([word for word in D2_m])

Train_m_2 = utils.concat_list([[eos_token + prpt + word for word in Train_m] for prpt in prompt_m])
Train_f_2 = utils.concat_list([[eos_token + prpt + word for word in Train_f] for prpt in prompt_f])
Test_m_2 = utils.concat_list([[eos_token + prpt + word for word in Test_m] for prpt in prompt_m])
Test_f_2 = utils.concat_list([[eos_token + prpt + word for word in Test_f] for prpt in prompt_f])

data_num = [1]*len(Train_m_1 + Train_f_1 + Test_m_1 + Test_f_1) + [2]*len(Train_m_2 + Train_f_2 + Test_m_2 + Test_f_2)
gender = [1]*len(Train_m_1) + [-1]*len(Train_f_1) + [1]*len(Test_m_1) + [-1]*len(Test_f_1) + [1]*len(Train_m_2) + [-1]*len(Train_f_2) + [1]*len(Test_m_2) + [-1]*len(Test_f_2)
train = [1]*len(Train_m_1 + Train_f_1) + [0]*len(Test_m_1 + Test_f_1) + [1]*len(Train_m_2 + Train_f_2) + [0]*len(Test_m_2 + Test_f_2)

D = {
  'examples': Train_m_1 + Train_f_1 + Test_m_1 + Test_f_1 + Train_m_2 + Train_f_2 + Test_m_2 + Test_f_2,
  'data_num': data_num,
  'gender': gender,
  'train': train,
  }

DF = pd.DataFrame.from_dict(D, orient = 'columns')
pd.DataFrame.to_csv(DF, "finding_gender_direction/Train_Data.csv", quotechar='"')