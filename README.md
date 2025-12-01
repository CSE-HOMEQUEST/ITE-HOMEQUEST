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
## 1. Choice of Algorithms

The HomeQuest dataset contains both categorical features (weekday, userId, category, mode, device type) and numerical features (points, energy usage, notification time).

User behavior also shows non-linear patterns, especially in time-of-day and task-type differences.

For these reasons, a tree-based ensemble model was selected.

We adopt the Gradient Boosting Classifier (GBDT) because it:

- handles mixed feature types without heavy preprocessing
- captures complex interactions in user activity patterns
- provides stable performance even with limited feature engineering

Two separate GBDT models are trained:

- a model predicting whether a daily/monthly challenge will be completed
- a model predicting whether a speed challenge will be completed within one hour

---

## 2. Features and Code Structure

### Feature Design

Each model uses event-level features extracted from the simulated logs.

User and Time Features

- weekday: weekly behavior patterns
- userId: differences in lifestyle rhythms among users
- notificationTime: key variable for speed-challenge responsiveness
- energyKwh: signals related to heating/energy tasks

Challenge Attributes

- category, mode
- durationType: distinguishes between short-term and long-term difficulty
- progressType: counter/device/energy, representing the nature of the action
- deviceType: appliance involvement

All categorical variables are transformed through one-hot encoding before training.

---

## 3. Code Structure Overview

The recommendation pipeline is structured as follows:

Dataset Loading

- A fixed six-month event log (CSV) is loaded and treated as the “historical behavior” of the family.
- This approach mirrors real smart-home environments where past interactions remain constant.

Model Training

- A time-aware split using `day_index` separates training and testing periods.
- Two GBDT models are trained: one for daily/monthly challenges and one for speed tasks.

Candidate Filtering

- Challenges with active cooldown restrictions are removed.
- Energy-saving tasks are considered only when heating usage increases.

Model Scoring

- The main model outputs completion probabilities.
- The speed model outputs one-hour completion probabilities for each possible notification time.

Diversity Adjustment

- Only the top candidates (top-K) are kept.
- A softmax-based sampling step prevents the system from recommending the same challenge repeatedly.
- Recommendation counts are stored in a JSON file and used to reduce the score of frequently recommended items.

Final Recommendation

- The system outputs one recommended daily task, one monthly task (if applicable),
    
    and one speed challenge with an optimized notification time.
    

This hybrid design combines model-based prediction with policy-based logic, enabling personalized and context-aware challenge recommendations.

## IV. Evaluation & Analysis
This section presents the evaluation of the Gradient Boosting Decision Tree model trained on six months of simulated HomeQuest activity logs. The goal is not perfect binary prediction but estimating relative completion likelihoods for ranking daily challenges.

A. Dataset and Evaluation Setup

Input features include weekday, personal points, family points, energy usage, mode, category, duration type, progress type, device type, and user identifier.

The target output is the completion of each challenge event.

Data was split chronologically to reflect real-world usage.

First two-thirds of the timeline used for training.

Remaining one-third used for testing.

This ensures that the model learns from past events and predicts future user behavior.

B. Main Model Quantitative Results

The main GBDT model produced the following scores on the test set:

AUC: 0.5587

RMSE: 0.547

MAE: 0.4758

Accuracy: 0.5574

Interpretation of these values:

User behavior in the dataset fluctuates heavily between zero and one.

Many important behavioral factors are not captured by available features.

Perfect classification is unrealistic in this setting.

The model is intended to generate relative probability scores for ranking challenges, not to make perfect binary predictions.

Therefore, these values are acceptable within the context of recommendation logic.

C. Trend Comparison Using Moving Average

To provide a clearer comparison between predicted probabilities and actual completion behavior, a moving average with a window size of ten samples was applied.

Insert Figure here:

Figure X. Smoothed Trends of Predicted and Actual Completion
(graph file: gbdt_smoothed_trend.png)

Use the following in Overleaf:

\begin{figure}[h]
\centering
\includegraphics[width=0.48\textwidth]{gbdt_smoothed_trend.png}
\caption{Smoothed trends of predicted probabilities and actual completion rates.}
\end{figure}

Interpretation:

The smoothed predicted probabilities rise during periods when the smoothed actual completion rate increases.

Both curves decrease during intervals of lower completion behavior.

The two curves share similar global patterns even though they do not match event by event.

This demonstrates that the model captures general behavioral tendencies despite high noise.

For a recommendation system, capturing trend-level information is more important than exact classification accuracy.

D. Speed Challenge Model

A separate GBDT model was trained to predict whether a speed-type challenge would be completed within one hour after notification. The model reached an AUC of 0.9994 in the simulated environment.

Interpretation:

The extremely high value is due to the deterministic structure of simulated duration data.

Training and testing were performed on the same dataset, creating an optimistic evaluation.

Real-world evaluation should include a proper time-based split.

E. Overall Interpretation

The main model functions as a ranking model rather than a strict classifier.

The moving average comparison confirms that the model captures meaningful behavioral patterns.

The model provides stable signals that are suitable for prioritizing challenge recommendations.

While the simulation contains noise and limited behavioral variables, the GBDT framework remains a practical foundation for future improvements.

## V. Related Work (e.g., existing studies)


## VI. Conclusion: Discussion
