# HomeQuest: AI-based Family Challenges with LG Devices
>한양대학교 2025-2 소프트웨어공학/인공지능및응용 프로젝트 (SWE/ITE Project in Hanyang Univ. 2025-2)


## Members
| Name | Organization | Email |
|------|-------------|--------|
| Hyeonseo Shim | Dept. of Information Systems, Hanyang University | andyandy21@hanyang.ac.kr |
| Sihyun Jang | Dept. of Information Systems, Hanyang University | jjj99nine2@hanyang.ac.kr |
| Jin Yun | Dept. of Information Systems, Hanyang University | jinijini0402@hanyang.ac.kr |
| Yeonsoo cho | Dept. of Public Administration, Hanyang University | meriel1010@hanyang.ac.kr |


## I. Introduction
### Motivation: Why are you doing this?

  We believe that smart homes should evolve beyond simply displaying appliance information and become an *experience platform* that naturally connects family members’ daily behaviors. However, current smart home services still follow a one-way structure—providing data and expecting users to act on their own—which makes it difficult to sustain real behavior change or meaningful family interaction.

  AI becomes essential in this project because it is the only technology capable of transforming the vast IoT and wearable data collected from everyday life into personalized challenge experiences. By learning each family’s life patterns, device-usage patterns, time preferences, and behavioral changes, AI can offer sustainable and customized challenges, naturally facilitate interactions among family members, and generate motivating feedback based on their progress in real time.

  With this approach, the smart home can move beyond a passive information system and become an interactive space where families naturally participate and engage with one another. In this environment, AI helps transform small everyday behaviors into shared moments of motivation, fun, and challenge—making the overall experience more meaningful and easier to sustain. This provides the foundation for what we aim to achieve with the HomeQuest project.

### what do you want to see at the end?
  We envision a smart home that goes beyond simply delivering information—a platform that seamlessly connects family members’ behaviors, habits, and meaningful changes.

  The ultimate goal of this project is to build an interactive challenge and feedback environment where AI analyzes IoT and wearable data to support shared participation and communication within the family.

  Through this system, we aim to create a smart-home experience in which small daily actions become meaningful motivation and enjoyable moments for every family member, ultimately transforming the home into a space that encourages collaboration, engagement, and positive routines.


## II. Datasets
The HomeQuest project constructs a dataset based on event-level simulation logs, designed to emulate real service conditions. Each time a family member interacts with or attempts a challenge, the system generates an event record containing contextual information such as challenge attributes, temporal features, device metadata, user identity, accumulated points, and the final outcome (success or failure).

This dataset structure enables the ML model to predict the likelihood of challenge completion under various conditions.

---

## 1. Dataset Structure

The dataset consists of several thousand event records, where each entry corresponds to a single challenge attempt.

Each event contains three main categories of information:

---

### 1) Challenge Attributes

- category: Type of challenge (household, energy, wellness)
- mode: daily / speed / monthly
- durationType: daily, weekly, or monthly period
- progressType: individual, family, or relay
- deviceType: Device involved (washer, air conditioner, smartwatch, etc.)

---

### 2) Temporal & Contextual Information

- eventDate: Date of the event
- day_index: Normalized day index within the dataset (for time-series analysis)
- weekday: Day of the week
- notificationTime: Time when the challenge notification was delivered
- completionTime: Time when the challenge was actually completed
- timeSlot: Time-of-day category (morning / afternoon / evening)

---

### 3) Outcome Information (Model Target Included)

- completed: Whether the challenge was completed (True/False) → model label (Y)
- personalPoints: Points earned by the user
- familyPoints: Points added to the family total
- energyKwh: Energy usage (for energy-related challenges)

---

### + Metadata (Identifiers)

- eventId, familyId, userId, challengeId

---

## 2. Purpose of the Dataset

The event-based dataset serves two primary purposes:

### 1) ML Model Training

Using the challenge attributes, temporal features, and contextual information,

the model is trained to predict whether a given event (challenge attempt) will be completed successfully.

### 2) Model Performance Evaluation

A subset of simulated events is used as the test set for evaluating the model’s performance using AUC (Area Under the ROC Curve).

This approach allows the model to learn realistic behavioral patterns even before real service data becomes available.

---

## 3. Sample Format (Actual Model Input)

```json
{
  "eventId": 10291,
  "familyId": "F001",
  "userId": "mom",
  "challengeId": "CH12",

  "category": "household",
  "mode": "daily",
  "durationType": "daily",
  "progressType": "individual",
  "deviceType": "washer",

  "eventDate": "2025-11-20",
  "day_index": 18,
  "weekday": 4,
  "notificationTime": "20:00",
  "completionTime": "21:13",
  "timeSlot": "evening",

  "completed": true,
  "personalPoints": 4,
  "familyPoints": 4,
  "energyKwh": 0.2
}

```

## III. Methodology
This project utilizes simulation-generated event data to train two classification models:

1. A general challenge completion prediction model, and
2. A speed-challenge model that predicts whether a challenge will be completed within one hour.

Both models use scikit-learn’s GradientBoostingClassifier (GBDT) with default hyperparameters, and only `random_state=42` is specified for reproducibility.

---

## 1. Algorithm Selection (Gradient Boosting Classifier)

GBDT is an ensemble method based on decision trees and is well suited for datasets like ours, which contain a mixture of categorical features (e.g., mode, category, timeSlot) and numerical features (e.g., points, energyKwh).

Tree-based models naturally capture combinational conditions, such as:

- “weekday + timeSlot + challenge category”
- “point level + mode type”

Additionally, GBDT provides probability outputs through `predict_proba()`, which makes it highly suitable for future extensions into a recommendation system that prioritizes challenges with a higher predicted success likelihood.

For these reasons, GBDT was selected as the core algorithm for this project.

---

## 2. Main Model: Predicting Challenge Completion (completed)

The first model predicts whether a challenge is completed (0/1) across all modes, including daily, speed, and monthly events.

### (1) Feature Construction

The following features from the codebase are used:

- weekday
- personalPoints
- familyPoints
- energyKwh
- timeSlot
- mode
- category
- durationType
- progressType
- deviceType

Categorical variables such as `timeSlot`, `mode`, and `category` are converted into one-hot encoded vectors before being fed into the model.

### (2) Training and Evaluation

To respect the time-dependent nature of the data,

the dataset is divided using `day_index`:

- first 2/3 → train set
- last 1/3 → test set

The model is trained using the default GBDT configuration, and predictions are evaluated using `predict_proba()`.

Model performance is assessed using AUC (ROC-AUC), which measures how well the model distinguishes between successful and unsuccessful challenge events.

---

## 3. Speed-Challenge Model: Predicting 1-Hour Completion (completed_within_1h)

The second model focuses exclusively on speed-mode events, predicting whether a speed challenge is completed within one hour after notification.

### (1) Label Construction

Notification and completion times are converted into minutes to compute `duration_min`.

The label `completed_within_1h` is defined as:

- completed = 1
- duration_min ≤ 60
    
    → `1`
    

Otherwise: `0`.

### (2) Feature Construction and Training

Features used for the speed model:

- weekday
- timeSlot
- category
- challengeId
- personalPoints
- familyPoints
- energyKwh

Categorical variables (timeSlot, category, challengeId) are one-hot encoded, and the entire set of speed events is used to train a GBDT classifier.

AUC is again used as the evaluation metric.

### (3) Practical Use

The trained speed model is used to estimate the probability of completing a speed challenge within one hour for different timeSlot options.

This allows the system to select and recommend the optimal timeSlot with the highest predicted success probability.

## IV. Evaluation & Analysis


## V. Related Work (e.g., existing studies)


## VI. Conclusion: Discussion
