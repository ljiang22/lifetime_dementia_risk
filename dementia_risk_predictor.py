import streamlit as st
import pandas as pd
import numpy as np
from mllib.bh_models import bh_model


def main():
    st.title('Lifetime Dementia Risk Predictor')
    st.write('(Tracking and preventing dementia before it gets too late!)')
    st.write('Dementia is a broad category of brain diseases that cause a long-term and often gradual decrease in the '
             'ability to think and remember that is severe enough to affect daily functioning. Other common symptoms include '
             'emotional problems, difficulties with language, and a decrease in motivation.'
             ' Consciousness is usually not affected. A diagnosis of dementia requires a change from a persons usual mental functioning and '
             'a greater decline than one would expect due to aging. (Wikipedia)')

    st.sidebar.title("User type")
    app_mode = st.sidebar.selectbox("Choose the user type",
        ["Common users", 'Use preexisting samples'])

    if app_mode == 'Use preexisting samples':
        sample_app()
    else:
        users_app()


def sample_app():
    sample_id = st.sidebar.number_input(label="Please input a sample number (0 -4600) (e.g. 0, 722, 929)", min_value=0,
                                        max_value=4600, value=0, step=1, format='%d')

    @st.cache
    def data_load():
        input_file = './raw data_edit/data_merge.csv'  # The well name of an input file
        data_input_ori = pd.read_csv(input_file)
        return data_input_ori
    data_input_ori = data_load()
    user1 = data_input_ori.loc[sample_id]
    user2 = user1[['Age', 'apoe', 'M/F', 'Hand', 'Education', 'Race', 'LIVSIT', 'RESIDENC', 'MARISTAT',
        'BMI', 'dem_idx', 'CVHATT', 'CVAFIB', 'CVANGIO', 'CVBYPASS', 'CVPACE', 'CVCHF', 'CVOTHR', 'CBSTROKE', 'CBTIA',
        'CBOTHR', 'PD', 'PDOTHR', 'SEIZURES', 'TRAUMBRF', 'TRAUMEXT', 'TRAUMCHR', 'NCOTHR', 'HYPERTEN', 'HYPERCHO',
        'DIABETES', 'B12DEF', 'THYROID', 'INCONTU', 'INCONTF', 'DEP2YRS', 'DEPOTHR', 'ALCOHOL', 'TOBAC30', 'TOBAC100',
        'SMOKYRS', 'PACKSPER', 'ABUSOTHR', 'PSYCDIS', 'GDS']]

    user2 = user2.sort_values(axis=0, ascending=False)
    user2['Age'] = np.floor(user2.Age)
    user2['BMI'] = np.floor(user2.BMI)
    #user2 = user2.drop(columns=['sumbox'])
    st.write('# The subject information is:', user2)

    input_data = user1
    thd = 50.0
    sumbox_max = 18.0
    model = bh_model(sumbox_max=sumbox_max, thd=thd)
    #st.write(input_data.shape)
    bh_score = model.bh_score(input_data)
    feed_back, score_pred, diff = model.feed_back(input_data)
    age_out, age_axis, score_age = model.age_analysis()
    '# The Evaluation Result is:'

    st.write('The subject brain health score is:', ('%3.1f') %bh_score)
    st.write('The evaluation result is:', feed_back)

    score_age1 = []
    for i in range(len(score_age)):
        score_tmp = float(score_age[i])
        #st.write(i, score_tmp)
        score_age1.append(score_tmp)
    score_age1 = np.asarray(score_age1)
    #score_age = np.reshape(score_age, (len(score_age)))
    #st.write(score_age1)
    #st.write('The age analysis:', age_axis, score_age)

    #with st.echo(code_location='below'):  # Put the code below the plot if necessary
    import matplotlib.pyplot as plt
    font = {'family': 'normal', 'size': 14}
    plt.rc('font', **font)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(age_axis, score_age1)
    ax.locator_params(axis='x', nbins=9)
    ax.set_xlabel("Age")
    ax.set_ylabel("Brain health score")

    st.write(fig)

    if type(age_out) == str:
        st.write(age_out)
    else:
        st.write(age_out[0], ('%3.0f') %age_out[1])

def users_app():
    """Diagnostic Model:
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
                           16) Sum all circled answers for a Total GDS Score  (GDS)"""


    st.sidebar.markdown("# Input your personal infos")
    #Age = st.sidebar.slider("What is your age?", 30, 111, 60) label="What is your age?",  min_value=30.0, step=1
    age = st.sidebar.number_input(label="What is your age? (30 - 110)", min_value=30, max_value=110, value=65, step=1, format='%d')

    height = st.sidebar.number_input(label="What is your height (inch)", min_value=20.0, max_value=120.0, value=70.0,
                                  step=1.0, format='%f')

    weight = st.sidebar.number_input(label="What is your weight (pounds)", min_value=50.0, max_value=350.0, value=250.0,
                                  step=1.0, format='%f')


    BMI = weight * 0.453592 / (height * 0.0254) ** 2.0  # w / h2, unit: (kg/m2)

    GDS = st.sidebar.number_input(label="What is your GDS score? (0-15) (Behavioral Assessment)", min_value=0.0, max_value=15.0, value=2.0,
                                  step=1.0, format='%f')

    SMOKYRS = st.sidebar.number_input(label="Total years smoked?", min_value=0.0, max_value=100.0, value=30.0,
                            step=1.0, format='%f')

    gene = st.sidebar.selectbox("Input your gene type", ['ε4', 'ε3','ε2', 'Unknown'])
    if gene == 'ε2':
        gene = 2.0
    elif gene == 'ε3':
        gene = 1.0
    elif gene == 'ε4':
        gene = 0.0
    else:
        gene = 1.0


    HYPERTEN = st.sidebar.selectbox("Hypertension?", ['Absent', 'Remote/Inactive','Recent/Active', 'Unknown'], index=2)
    if HYPERTEN == 'Absent':
        HYPERTEN = 0.0
    elif HYPERTEN == 'Remote/Inactive':
        HYPERTEN = 1.0
    elif HYPERTEN == 'Recent/Active':
        HYPERTEN = 2.0
    else:
        HYPERTEN = 0.0


    edu = st.sidebar.number_input(label="What is your education? High school/GED = 12, Bachelors degree = 16, Master’s degree = 18, Doctorate = 20",
                            min_value=7, max_value=29, value=15, step=1, format='%d')

    gender = st.sidebar.selectbox("What is your gender? ", ['Male', 'Female'])
    if gender == 'Male':
        gender = 1.0
    else:
        gender = 0.0

    hand = st.sidebar.selectbox("What is your handedness? ", ['Left', 'Right', 'Ambidextrous'])
    if hand == 'Right':
        hand = 2.0
    elif hand == 'Left':
        hand = 1.0
    else:
        hand = 0.0

    maris = st.sidebar.selectbox("What is your marital status? ", ['Married', 'Widowed', 'Divorced', 'Separated', 'Never married', 'Living as married', 'Other'])
    if maris == 'Married':
        maris = 0.0
    elif maris == 'Living as married':
        maris = 1.0
    elif maris == 'Widowed':
        maris = 2.0
    elif maris == 'Divorced':
        maris = 3.0
    elif maris == 'Separated':
        maris = 4.0
    elif maris == 'Never married':
        maris = 5.0
    else:
        maris = 2.0  # Treat 'others' as a medium value for this variable. May need more consideration

    livsit = st.sidebar.selectbox("What is your living sitution? ",
                                 ['Lives with spouse or partner', 'Lives with relative or friend', 'Lives with group', 'Lives alone', 'Other'])
    if livsit == 'Lives with spouse or partner':
        livsit = 0.0
    elif livsit == 'Lives with relative or friend':
        livsit = 1.0
    elif livsit == 'Lives with group':
        livsit = 2.0
    elif livsit == 'Lives alone':
        livsit = 3.0
    else:
        livsit = 4.0

    residenc = st.sidebar.selectbox("What is the type of residence? ",
                                 ['Single family residence', 'Retirement community', 'Assisted living/boarding home/adult family home',
                                  'Skilled nursing facility/nursing home', 'Other'])
    if residenc == 'Single family residence':
        residenc = 0.0
    elif residenc == 'Retirement community':
        residenc = 1.0
    elif residenc == 'Assisted living/boarding home/adult family home':
        residenc = 2.0
    elif residenc == 'Skilled nursing facility/nursing home':
        residenc = 3.0
    else:
        residenc = 4.0

    fmlh = st.sidebar.selectbox("How many of your family members have dementia?",
                                 [0, 1, 2, 3, 4, 5, 6], index=3)
    dem_idx = fmlh


    CVHATT = st.sidebar.selectbox("Heart attack/cardiac arrest?", ['Absent', 'Remote/Inactive','Recent/Active', 'Unknown'])
    if CVHATT == 'Absent':
        CVHATT = 0.0
    elif CVHATT == 'Remote/Inactive':
        CVHATT = 1.0
    elif CVHATT == 'Recent/Active':
        CVHATT = 2.0
    else:
        CVHATT = 0.0

    CVAFIB = st.sidebar.selectbox("Atrial fibrillation?", ['Absent', 'Remote/Inactive','Recent/Active', 'Unknown'])
    if CVAFIB == 'Absent':
        CVAFIB = 0.0
    elif CVAFIB == 'Remote/Inactive':
        CVAFIB = 1.0
    elif CVAFIB == 'Recent/Active':
        CVAFIB = 2.0
    else:
        CVAFIB = 0.0

    CVANGIO = st.sidebar.selectbox("Angioplasty/endarterectomy/stent?", ['Absent', 'Remote/Inactive','Recent/Active', 'Unknown'])
    if CVANGIO == 'Absent':
        CVANGIO = 0.0
    elif CVANGIO == 'Remote/Inactive':
        CVANGIO = 1.0
    elif CVANGIO == 'Recent/Active':
        CVANGIO = 2.0
    else:
        CVANGIO = 0.0

    CVBYPASS = st.sidebar.selectbox("Cardiac bypass procedure?", ['Absent', 'Remote/Inactive','Recent/Active', 'Unknown'])
    if CVBYPASS == 'Absent':
        CVBYPASS = 0.0
    elif CVBYPASS == 'Remote/Inactive':
        CVBYPASS = 1.0
    elif CVBYPASS == 'Recent/Active':
        CVBYPASS = 2.0
    else:
        CVBYPASS = 0.0

    CVPACE = st.sidebar.selectbox("Pacemaker?", ['Absent', 'Remote/Inactive','Recent/Active', 'Unknown'])
    if CVPACE == 'Absent':
        CVPACE = 0.0
    elif CVPACE == 'Remote/Inactive':
        CVPACE = 1.0
    elif CVPACE == 'Recent/Active':
        CVPACE = 2.0
    else:
        CVPACE = 0.0

    CVCHF = st.sidebar.selectbox("Congestive heart failure?", ['Absent', 'Remote/Inactive','Recent/Active', 'Unknown'])
    if CVCHF == 'Absent':
        CVCHF = 0.0
    elif CVCHF == 'Remote/Inactive':
        CVCHF = 1.0
    elif CVCHF == 'Recent/Active':
        CVCHF = 2.0
    else:
        CVCHF = 0.0

    CVOTHR = st.sidebar.selectbox("Cardiovascular disease, other?", ['Absent', 'Remote/Inactive','Recent/Active', 'Unknown'])
    if CVOTHR == 'Absent':
        CVOTHR = 0.0
    elif CVOTHR == 'Remote/Inactive':
        CVOTHR = 1.0
    elif CVOTHR == 'Recent/Active':
        CVOTHR = 2.0
    else:
        CVOTHR = 0.0

    CBSTROKE = st.sidebar.selectbox("Stroke?", ['Absent', 'Remote/Inactive','Recent/Active', 'Unknown'])
    if CBSTROKE == 'Absent':
        CBSTROKE = 0.0
    elif CBSTROKE == 'Remote/Inactive':
        CBSTROKE = 1.0
    elif CBSTROKE == 'Recent/Active':
        CBSTROKE = 2.0
    else:
        CBSTROKE = 0.0

    CBTIA = st.sidebar.selectbox("Transient ischemic attack?", ['Absent', 'Remote/Inactive','Recent/Active', 'Unknown'])
    if CBTIA == 'Absent':
        CBTIA = 0.0
    elif CBTIA == 'Remote/Inactive':
        CBTIA = 1.0
    elif CBTIA == 'Recent/Active':
        CBTIA = 2.0
    else:
        CBTIA = 0.0

    CBOTHR = st.sidebar.selectbox("Cerebrovascular disease, other?", ['Absent', 'Remote/Inactive','Recent/Active', 'Unknown'])
    if CBOTHR == 'Absent':
        CBOTHR = 0.0
    elif CBOTHR == 'Remote/Inactive':
        CBOTHR = 1.0
    elif CBOTHR == 'Recent/Active':
        CBOTHR = 2.0
    else:
        CBOTHR = 0.0

    PD = st.sidebar.selectbox("Parkinson’s disease?", ['Absent', 'Remote/Inactive','Recent/Active', 'Unknown'])
    if PD == 'Absent':
        PD = 0.0
    elif PD == 'Remote/Inactive':
        PD = 1.0
    elif PD == 'Recent/Active':
        PD = 2.0
    else:
        PD = 0.0

    PDOTHR = st.sidebar.selectbox("Other Parkinsonism disorder?", ['Absent', 'Remote/Inactive','Recent/Active', 'Unknown'])
    if PDOTHR == 'Absent':
        PDOTHR = 0.0
    elif PDOTHR == 'Remote/Inactive':
        PDOTHR = 1.0
    elif PDOTHR == 'Recent/Active':
        PDOTHR = 2.0
    else:
        PDOTHR = 0.0

    SEIZURES = st.sidebar.selectbox("Seizures?", ['Absent', 'Remote/Inactive','Recent/Active', 'Unknown'])
    if SEIZURES == 'Absent':
        SEIZURES = 0.0
    elif SEIZURES == 'Remote/Inactive':
        SEIZURES = 1.0
    elif SEIZURES == 'Recent/Active':
        SEIZURES = 2.0
    else:
        SEIZURES = 0.0

    TRAUMBRF = st.sidebar.selectbox("Traumatic brain injury with brief loss of consciousness ( < 5 minutes)?",
                                    ['Absent', 'Remote/Inactive','Recent/Active', 'Unknown'])
    if TRAUMBRF == 'Absent':
        TRAUMBRF = 0.0
    elif TRAUMBRF == 'Remote/Inactive':
        TRAUMBRF = 1.0
    elif TRAUMBRF == 'Recent/Active':
        TRAUMBRF = 2.0
    else:
        TRAUMBRF = 0.0

    TRAUMEXT = st.sidebar.selectbox("Traumatic brain injury with extended loss of consciousness ( > 5minutes)?",
                                    ['Absent', 'Remote/Inactive','Recent/Active', 'Unknown'])
    if TRAUMEXT == 'Absent':
        TRAUMEXT = 0.0
    elif TRAUMEXT == 'Remote/Inactive':
        TRAUMEXT = 1.0
    elif TRAUMEXT == 'Recent/Active':
        TRAUMEXT = 2.0
    else:
        TRAUMEXT = 0.0

    TRAUMCHR = st.sidebar.selectbox("Traumatic brain injury with chronic deficit or dysfunction?", ['Absent', 'Remote/Inactive','Recent/Active', 'Unknown'])
    if TRAUMCHR == 'Absent':
        TRAUMCHR = 0.0
    elif TRAUMCHR == 'Remote/Inactive':
        TRAUMCHR = 1.0
    elif TRAUMCHR == 'Recent/Active':
        TRAUMCHR = 2.0
    else:
        TRAUMCHR = 0.0

    NCOTHR = st.sidebar.selectbox("Other neurologic conditions, other?", ['Absent', 'Remote/Inactive','Recent/Active', 'Unknown'])
    if NCOTHR == 'Absent':
        NCOTHR = 0.0
    elif NCOTHR == 'Remote/Inactive':
        NCOTHR = 1.0
    elif NCOTHR == 'Recent/Active':
        NCOTHR = 2.0
    else:
        NCOTHR = 0.0

    HYPERCHO = st.sidebar.selectbox("Hypercholesterolemia?", ['Absent', 'Remote/Inactive','Recent/Active', 'Unknown'])
    if HYPERCHO == 'Absent':
        HYPERCHO = 0.0
    elif HYPERCHO == 'Remote/Inactive':
        HYPERCHO = 1.0
    elif HYPERCHO == 'Recent/Active':
        HYPERCHO = 2.0
    else:
        HYPERCHO = 0.0

    DIABETES = st.sidebar.selectbox("Diabetes?", ['Absent', 'Remote/Inactive','Recent/Active', 'Unknown'])
    if DIABETES == 'Absent':
        DIABETES = 0.0
    elif DIABETES == 'Remote/Inactive':
        DIABETES = 1.0
    elif DIABETES == 'Recent/Active':
        DIABETES = 2.0
    else:
        DIABETES = 0.0

    B12DEF = st.sidebar.selectbox("B12 deficiency?", ['Absent', 'Remote/Inactive','Recent/Active', 'Unknown'])
    if B12DEF == 'Absent':
        B12DEF = 0.0
    elif B12DEF == 'Remote/Inactive':
        B12DEF = 1.0
    elif B12DEF == 'Recent/Active':
        B12DEF = 2.0
    else:
        B12DEF = 0.0

    THYROID = st.sidebar.selectbox("Thyroid disease?", ['Absent', 'Remote/Inactive','Recent/Active', 'Unknown'])
    if THYROID == 'Absent':
        THYROID = 0.0
    elif THYROID == 'Remote/Inactive':
        THYROID = 1.0
    elif THYROID == 'Recent/Active':
        THYROID = 2.0
    else:
        THYROID = 0.0

    INCONTU = st.sidebar.selectbox("Incontinence – urinary?", ['Absent', 'Remote/Inactive','Recent/Active', 'Unknown'])
    if INCONTU == 'Absent':
        INCONTU = 0.0
    elif INCONTU == 'Remote/Inactive':
        INCONTU = 1.0
    elif INCONTU == 'Recent/Active':
        INCONTU = 2.0
    else:
        INCONTU = 0.0

    INCONTF = st.sidebar.selectbox("Incontinence – bowel?", ['Absent', 'Remote/Inactive','Recent/Active', 'Unknown'])
    if INCONTF == 'Absent':
        INCONTF = 0.0
    elif INCONTF == 'Remote/Inactive':
        INCONTF = 1.0
    elif INCONTF == 'Recent/Active':
        INCONTF = 2.0
    else:
        INCONTF = 0.0

    DEP2YRS = st.sidebar.selectbox("Depression, active within the past 2 years?", ['Absent', 'Remote/Inactive','Recent/Active', 'Unknown'])
    if DEP2YRS == 'Absent':
        DEP2YRS = 0.0
    elif DEP2YRS == 'Remote/Inactive':
        DEP2YRS = 1.0
    elif DEP2YRS == 'Recent/Active':
        DEP2YRS = 2.0
    else:
        DEP2YRS = 0.0

    DEPOTHR = st.sidebar.selectbox("Depression, other episodes (prior to 2 years)?", ['Absent', 'Remote/Inactive','Recent/Active', 'Unknown'])
    if DEPOTHR == 'Absent':
        DEPOTHR = 0.0
    elif DEPOTHR == 'Remote/Inactive':
        DEPOTHR = 1.0
    elif DEPOTHR == 'Recent/Active':
        DEPOTHR = 2.0
    else:
        DEPOTHR = 0.0

    ALCOHOL = st.sidebar.selectbox("Substance abuse – alcohol?", ['Absent', 'Remote/Inactive','Recent/Active', 'Unknown'])
    if ALCOHOL == 'Absent':
        ALCOHOL = 0.0
    elif ALCOHOL == 'Remote/Inactive':
        ALCOHOL = 1.0
    elif ALCOHOL == 'Recent/Active':
        ALCOHOL = 2.0
    else:
        ALCOHOL = 0.0

    TOBAC30 = st.sidebar.selectbox("Cigarette smoking history – Has subject smoked within last 30 days?", ['Yes', 'No'])
    if TOBAC30 == 'No':
        TOBAC30 = 0.0
    else:
        TOBAC30 = 1.0

    TOBAC100 = st.sidebar.selectbox("Cigarette smoking history - Has subject smoked more than 100 cigarettes in his/her life?", ['Yes', 'No'])
    if TOBAC100 == 'No':
        TOBAC100 = 0.0
    else:
        TOBAC100 = 1.0


    PACKSPER = st.sidebar.number_input(label="Average number of packs/day smoked?", min_value=0.0, max_value=5.0, value=1.0,
                            step=0.5, format='%f')
    if PACKSPER < 0.5:
        PACKSPER = 1.0
    elif PACKSPER >= 0.5 and PACKSPER < 1.0:
        PACKSPER = 2.0
    elif PACKSPER >= 1.0 and PACKSPER < 1.5:
        PACKSPER = 3.0
    elif PACKSPER >= 1.5 and PACKSPER < 2.0:
        PACKSPER = 4.0
    elif PACKSPER >= 2.0:
        PACKSPER = 5.0

    ABUSOTHR = st.sidebar.selectbox("Other abused substances?", ['Absent', 'Remote/Inactive','Recent/Active', 'Unknown'])
    if ABUSOTHR == 'Absent':
        ABUSOTHR = 0.0
    elif ABUSOTHR == 'Remote/Inactive':
        ABUSOTHR = 1.0
    elif ABUSOTHR == 'Recent/Active':
        ABUSOTHR = 2.0
    else:
        ABUSOTHR = 0.0

    PSYCDIS = st.sidebar.selectbox("Psychiatric disorders?", ['Absent', 'Remote/Inactive','Recent/Active', 'Unknown'])
    if PSYCDIS == 'Absent':
        PSYCDIS = 0.0
    elif PSYCDIS == 'Remote/Inactive':
        PSYCDIS = 1.0
    elif PSYCDIS == 'Recent/Active':
        PSYCDIS = 2.0
    else:
        PSYCDIS = 0.0

    race = 1.0

    st.sidebar.markdown('# Clinical Dementia Rating')

    memory = st.sidebar.number_input(label="What is your memory rating? (0-3)", min_value=0.0, max_value=3.0, value=1.5,
                                  step=0.5, format='%f')
    commun = st.sidebar.number_input(label="What is your community affair rating? (0-3)", min_value=0.0, max_value=3.0, value=1.0,
                                  step=0.5, format='%f')
    homehobb = st.sidebar.number_input(label="What is your home & hobbies rating? (0-3)", min_value=0.0, max_value=3.0, value=1.0,
                                  step=0.5, format='%f')
    judgment = st.sidebar.number_input(label="What is your judgement rating? (0-3)", min_value=0.0, max_value=3.0, value=1.0,
                                  step=0.5, format='%f')
    orient = st.sidebar.number_input(label="What is your orientation rating? (0-3)", min_value=0.0, max_value=3.0, value=1.0,
                                  step=0.5, format='%f')
    perscare = st.sidebar.number_input(label="What is your personal care rating? (0-3)", min_value=0.0, max_value=3.0, value=1.0,
                                  step=0.5, format='%f')
    cdr_clc = memory + commun + homehobb + judgment + orient + perscare
    #st.write('# The standard clinical dementia rating (CDR) is:', cdr_clc)

    user_info_ori = [commun, homehobb, judgment, memory, orient, perscare, cdr_clc, age, gene, gender, hand, edu, race, livsit, residenc, maris, BMI, dem_idx, CVHATT, CVAFIB, CVANGIO, CVBYPASS, CVPACE, CVCHF, CVOTHR, CBSTROKE, CBTIA, CBOTHR, PD, PDOTHR, SEIZURES, TRAUMBRF, TRAUMEXT, TRAUMCHR, NCOTHR, HYPERTEN, HYPERCHO, DIABETES, B12DEF, THYROID, INCONTU, INCONTF, DEP2YRS, DEPOTHR, ALCOHOL, TOBAC30, TOBAC100, SMOKYRS, PACKSPER, ABUSOTHR, PSYCDIS, GDS]
    user_info_ori = np.reshape(user_info_ori, (1, len(user_info_ori)))
    keys = ['commun', 'homehobb', 'judgment', 'memory', 'orient', 'perscare', 'sumbox', 'Age', 'apoe', 'M/F', 'Hand', 'Education', 'Race', 'LIVSIT', 'RESIDENC', 'MARISTAT',
     'BMI', 'dem_idx', 'CVHATT', 'CVAFIB', 'CVANGIO', 'CVBYPASS', 'CVPACE', 'CVCHF', 'CVOTHR', 'CBSTROKE', 'CBTIA',
     'CBOTHR', 'PD', 'PDOTHR', 'SEIZURES', 'TRAUMBRF', 'TRAUMEXT', 'TRAUMCHR', 'NCOTHR', 'HYPERTEN', 'HYPERCHO',
     'DIABETES', 'B12DEF', 'THYROID', 'INCONTU', 'INCONTF', 'DEP2YRS', 'DEPOTHR', 'ALCOHOL', 'TOBAC30', 'TOBAC100',
     'SMOKYRS', 'PACKSPER', 'ABUSOTHR', 'PSYCDIS', 'GDS']


    #st.write(len(user_info_ori), len(keys))

    user_info_ori = pd.DataFrame(user_info_ori, columns=keys)
    user_info = user_info_ori.drop(columns=['sumbox','commun', 'homehobb', 'judgment', 'memory', 'orient', 'perscare'] )

    user_sort = user_info.sort_values(by=[0], axis=1, ascending=False)
    user_sort['BMI'] = np.floor(BMI)
    #st.write('# Your input information is:', user_sort)

    input_data = user_info_ori
    thd = 50.0
    sumbox_max = 18.0
    model = bh_model(sumbox_max=sumbox_max, thd=thd)
    bh_score = model.bh_score(input_data)
    feed_back, score_pred, diff = model.feed_back(input_data)
    age_out, age_axis, score_age = model.age_analysis()

    risk_factors = model.risk_factor()
    '# The Evaluation Result is:'

    st.write('1) Your brain health score is:', ('%3.1f') %bh_score)
    st.write('2) The evaluation result is:', feed_back)

    score_age1 = []
    for i in range(len(score_age)):
        score_tmp = float(score_age[i])
        #st.write(i, score_tmp)
        score_age1.append(score_tmp)
    score_age1 = np.asarray(score_age1)
    #score_age = np.reshape(score_age, (len(score_age)))
    #st.write(score_age1)
    #st.write('The age analysis:', age_axis, score_age)

    # BMI
    #output_factors, score_factors = model.risk_factor()
    #output_bmi, score_bmi = model.risk_bmi()
    #output_edu, score_edu = model.risk_edu()
    #output_gds, score_gds = model.risk_gds()

    #with st.echo(code_location='below'):  # Put the code below the plot if necessary
    if type(age_out) == str:
        st.write(age_out)
    else:
        st.write('3)', age_out[0], ('%3.0f') %age_out[1])

    st.write('4) The top risk factors for you are:')
    for i in range(len(risk_factors)):
        st.write(risk_factors[i])

    import matplotlib.pyplot as plt
    font = {'family': 'normal', 'size': 14}
    plt.rc('font', **font)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(age_axis, score_age1, color='black')
    ax.locator_params(axis='x', nbins=9)
    ax.set_xlabel("Age")
    ax.set_ylabel("Brain health score")

    st.write(fig)
    st.write('(Normal: 80 – 100; Mild impairment: 60 – 80; Dementia: 0 – 60)')
    st.write('(Please note that the brain health score is calculated using the standard clinical dementia rating.)')


    #if type(output_bmi) != str:
        #st.write('If you take active preventions, you will have a lower potential dementia risk, at age:', ('%3.0f') %output_bmi[1])



if __name__ == "__main__":
    main()




















