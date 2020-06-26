import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import set_option
from mllib.preprocess import normalize
from mllib.preprocess import unnormalize
from mllib.networks import evaluate
from mllib.networks import MLP_train_opt
from mllib.networks import MLP_plot
from mllib.networks import MLP_REG_v1
from mllib.networks import MLPR_v0
from sklearn.svm import SVR
from sklearn.feature_selection import VarianceThreshold
import pickle
from mllib.bh_models import bh_model


set_option('display.width', 2000)
pd.set_option("display.max_rows", 500, "display.max_columns", 2000)
set_option('precision', 3)
#set_option("display.max_rows", 10)
pd.options.mode.chained_assignment = None

input_file = './raw data_edit/data_merge.csv'    # The well name of an input file
data_input_ori = pd.read_csv(input_file)
print(data_input_ori.head())
keys = data_input_ori.keys()
print(keys)

print('Please fill all the evaluation form provided! (Note: The more accurate the information you provide, the more accurate the result is!)')
"""
/**************************************************************************************************************/
Diagnostic Model:
1. Memory
2. Judgment
3. Orientation
4. Personal care
5. Community affair
6. Home  & hobbies
Standard Clinical Dementia Rating (SCDR)   

7. Positron emission tomography (PET) (optional)
8. Magnetic resonance imaging (MRI) (optional)
Comprehensive Assessment of Brain Functionalities. -- May be used to quantitatively describe the brain change. Could be more specific.

Evaluation Model:
Demographic:  1) age 2) race 3) gender 4) education 5) handedness 6) living situation 7) level of independence 
              8) type of residence 9) marital status 10) Body mass index (BMI)
Family history: if any of the subject's family members have been diagnosed as dementia?
Health history: 1) Heart attack/cardiac arrest (CVHATT) 2) Atrial fibrillation (CVAFIB) 3) Angioplasty/endarterectomy/stent (CVANGIO) 
                4) Cardiac bypass procedure (CVBYPASS) 5) Pacemaker (CVPACE) 6) Congestive heart failure (CVCHF) 7) Cardiovascular disease, other (CVOTHR)
                8) Stroke (CBSTROKE) 9) Transient ischemic attack (CBTIA) 10) Cerebrovascular disease, other (CBOTHR) 11) Parkinson’s disease (PD)
                12) Other Parkinsonism disorder (PDOTHR) 13) Seizures 14) Brain trauma – brief unconsciousness (TRAUMBRF) 
                15) Brain trauma – extended unconsciousness (TRAUMEXT)  16) Traumatic brain injury with chronic deficit or dysfunction (TRAUMCHR) 
                17) Other neurologic conditions, other (NCOTHR) 18) Hypertension (HYPERTEN)
                19) Hypercholesterolemia (HYPERCHO) 20) Diabetes 21) B12 deficiency (B12DEF) 22) Thyroid disease (THYROID) 23) Incontinence – urinary (INCONTU)
                24) Incontinence – bowel (INCONTF) 25) Depression, active within the past 2 years (DEP2YRS) 26) Depression, other episodes (DEPOTHR)
                27) ALCOHOL 28) Cigarette smoking history – last 30 days (TOBAC30) 29) Cigarette smoking history - 100 lifetime cigarettes (TOBAC100)
                30) Total years smoked (SMOKYRS) 31) Average number of packs/day smoked (PACKSPER) 32) Other abused substances (ABUSOTHR) 33) Psychiatric disorders (PSYCDIS)
Genotyping (Optional): The apolipoprotein E gene (APOE) has been linked to increased risk for Alzheimer’s disease, 
             while the ε2 allele (APOE ε2) may provide protection from Alzheimer’s disease
Behavioral assessment: 1) Satisfied with life (SATIS) 2) Dropped activities and interests (DROPACT) 3) Life feels empty (EMPTY) 
                       4) Bored (BORED) 5) Are you in good spirits most of the time? (SPIRITS) 6) Afraid bad thing will happen (AFRAID)
                       7) Do you feel happy most of the time? (HAPPY) 8) Feel helpless (HELPLESS) 9) Prefer to stay home (STAYHOME) 
                       10) Do you feel you have more problems with memory than most? (MEMPROB) 
                       11) Do you think it is wonderful to be alive now? (WONDRFUL) 12) Feel worthless (WRTHLESS)
                       13) Full of energy (ENERGY) 14) Do you feel that your situation is hopeless? (HOPELESS) 
                       15) Others are better off (Others are better off) 
                       16) Sum all circled answers for a Total GDS Score  (GDS) 
/**********************************************************************************************/                       
"""

user_id = 722  # 0, 722, 929      82, 1768, 692, 722

user1 = data_input_ori.loc[user_id]
print('The user information is:', user1)

input_data = user1
thd = 30.0
sumbox_max = 18.0
model = bh_model(sumbox_max=sumbox_max, thd=thd)
bh_score = model.bh_score(input_data)
feed_back, score_pred, diff = model.feed_back(input_data)
age_out = model.age_analysis()
model.risk_factor()
print('Your brain health score is:', bh_score)
print('The evaluation result is:', feed_back)
print('The age analysis:', age_out)



