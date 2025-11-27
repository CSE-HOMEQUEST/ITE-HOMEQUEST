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
HomeQuest uses a fixed six-month simulated event-log dataset designed to mimic a real smart-home service environment.

Each event represents a full challenge interaction by a family member and contains challenge attributes, user context, time information, device data, energy usage, and success outcomes.

This structure enables machine-learning models to learn how various conditions influence the likelihood of completing each challenge.

---

## Dataset Structure

The dataset consists of approximately 366 days of logs for a single family, with multiple events recorded each day.

Each event includes the following groups of information.

---

### Challenge Metadata

- `challengeId` — unique identifier for each challenge
- `category` — thematic group (health, cleaning, laundry, energy, etc.)
- `mode` — challenge type (daily task, speed challenge, monthly goal)
- `durationType` — expected duration (short or long)
- `progressType` — how progress is measured (counter/device/energy)
- `deviceType` — device involved (washer, AC, robot vacuum, none)
- `cooldown_days` — required waiting period before the challenge can be repeated

---

### Time, User, and Context Information

- `day_index` — sequential index of days for time-series learning
- `eventDate` — actual calendar date of the event
- `weekday` — day of week (0–6) representing user routine patterns
- `userId` — user identity, mapped to distinct lifestyle profiles
- `familyId` — household identifier (single family in demo)
- `notificationTime` — time when the challenge was delivered to the user
- `completionTime` — time when the challenge was completed (if any)
- `energyKwh` — daily heating/energy consumption related to energy tasks

Different users follow different lifestyle patterns (morning-type, regular, evening-type, student), allowing the model to learn variations in activity timing and behavior.

---

### Outcome Labels

Two labels are provided for model training.

completed

– 1 if the challenge was completed within the same day

– used as the target of the main model

completed_within_1h

– 1 if the user completed the challenge within 60 minutes after the notification

– used for the speed-challenge time optimization model

---

## Dataset Usage

The dataset is used for both training and evaluating machine-learning models.

---

### Model Training

Main Model

Predicts the probability of completing a challenge under the current conditions.

Inputs include weekday, category, mode, points, device type, userId, and contextual features.

Speed Model

Predicts the probability of completing a speed-mode challenge within one hour.

Uses weekday, notification time (in minutes), userId, and challenge attributes.

---

### Model Evaluation

A time-based train/test split is applied using `day_index`, and AUC is used as the evaluation metric.

Because the dataset is fixed, the evaluation is reproducible.

---

## Sample Record
| day_index | userId  | challengeId       | category     | mode     | notificationTime | completionTime | completed | personalPoints | energyKwh |
|-----------|---------|-------------------|--------------|----------|------------------|----------------|-----------|----------------|-----------|
| 12        | user_4  | daily_water_2     | health       | daily    | 09:00:00         | 09:12:00       | 1         | 4              | 0.0       |
| 12        | user_4  | speed_dishwasher  | dishwashing  | speed    | 18:00:00         | 18:34:00       | 1         | 4              | 0.0       |
| 30        | user_4  | monthly_heating   | energy       | monthly  | 20:00:00         | —              | 0         | 4              | 0.7       |




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
