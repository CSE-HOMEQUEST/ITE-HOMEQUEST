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

Initially, the dataset contained about 360 manually constructed events, but this volume was not sufficient for training robust machine-learning models.

To address this, we expanded the dataset to roughly 10,000 records by generating additional synthetic events using the OpenAI API, following the same schema and behavioral constraints as the original data.

Each event represents a full challenge interaction by a family member and contains challenge attributes, user context, time information, device data, energy usage, and success outcomes.

---

## Dataset Structure

The dataset consists of logs for a single family, with multiple events recorded each day.

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

Two outcome labels are used for model training:

completed
Indicates whether the challenge was finished on the same day (1 = completed).
This label serves as the target variable for the main completion model.

completed_within_1h
Indicates whether the user completed the challenge within 60 minutes after receiving the notification (1 = completed).
This label is used for the speed-challenge model, which predicts short-term completion behavior.

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

- Low preprocessing overhead

GBDT can handle both categorical and numerical variables without heavy transformation.
This keeps the pipeline simple and allows the augmented dataset—with many new combinations of feature values—to be used directly without extensive encoding.

- Effective at learning non-linear behavioral patterns

Challenge completion often depends on interactions among factors such as time of day, user type, and device usage.
GBDT captures these non-linear relationships naturally, enabling the model to represent realistic user behavior where multiple conditions interact simultaneously.

- Well-suited for probability-based recommendation

The HomeQuest recommendation logic begins by estimating the probability that each challenge will be completed under current conditions, then uses this probability as the ranking score.
GBDT, trained with logistic loss, produces stable probability estimates for binary outcomes, aligning perfectly with this workflow.
It also accommodates diverse behavioral contexts, enabling more accurate probability predictions that feed directly into Softmax sampling and the final recommendation step.

- Fast inference for real-time recommendation

Because prediction is based on tree traversal, inference is extremely fast.
Even with the dataset expanded to 10,000 records, the model remains lightweight enough to generate recommendations instantly whenever the user opens the app.

- High model interpretability

Feature importance analysis allows us to understand which factors influence completion likelihood.
This transparency is useful for system refinement and explaining recommendation outcomes to stakeholders.

---

## 2. Features design



- weekday

User activity patterns vary significantly across the week.
Workdays, weekends, and routine patterns tied to specific days strongly influence whether a user can complete a challenge, making this a key temporal feature.

- userId (user profile)

Each user follows a distinct lifestyle pattern—morning-type, evening-type, student, etc.—which affects their likelihood of completing certain tasks.
The same challenge can have very different success probabilities depending on the user, so this feature is essential.

- category (challenge category)

Health, household, and energy-saving challenges differ significantly in both difficulty and user motivation.
Because each category has its own baseline success pattern, this feature allows the model to capture the structural differences in how users respond to each type of task.

- mode (daily / speed / monthly)

Each challenge mode has unique completion conditions and user burden levels.
Speed challenges in particular have strict time constraints, creating distinct behavioral patterns the model must learn.

- deviceType

Some challenges require specific appliances such as a washer, AC, or robot vacuum.
Availability and typical usage patterns of these devices influence completion probability, making deviceType an important contextual input.

- notificationTime

The time at which a challenge is delivered greatly impacts success.
Earlier notifications generally lead to higher completion rates, while late notifications tend to decrease them, so this feature plays a key role in modeling user responsiveness.

- personalPoints (engagement indicator)

Higher accumulated points signal a user who consistently engages with challenges.
Users with high participation scores tend to complete more tasks, making this feature a strong predictor of success.

- energyKwh (daily energy consumption)

For energy-related challenges, the user’s actual consumption pattern is directly tied to task execution.
Heating or cooling challenges often correlate with daily energy usage levels, giving this feature predictive value.

- progressType / durationType / cooldown_days (contextual task properties)

These fields describe how a challenge is completed (counter-based, device-linked, energy-based), how long it typically takes, and whether a mandatory cooldown applies.
Such structural characteristics directly influence completion feasibility and thus contribute meaningfully to the model’s predictions.

## 3. Code Structure Overview

The HomeQuest recommendation engine follows a clear sequence:
(1) candidate filtering → (2) model scoring → (3) score adjustment & diversity → (4) final recommendation generation.

### 1) Candidate Filtering (Cooldown + Energy Condition)

The engine first removes challenges that cannot be recommended today.
Challenge metadata and cooldown rules determine whether each challenge is eligible.

```python
challenge_meta = pd.DataFrame([...], columns=[
    "challengeId", "category", "mode", "durationType",
    "progressType", "deviceType", "cooldown_days"
])

def is_available(ch, last_done, today):
    cd = ch["cooldown_days"]
    if cd == 0:
        return True
    cid = ch["challengeId"]
    if cid not in last_done:
        return True
    return (today - last_done[cid]) >= cd
```

Additionally, monthly heating-saving challenges are recommended only if recent heating usage has increased, based on a simple comparison of energy consumption in the earlier vs. later period.
```python
e = events[events["category"] == "energy"]
prev = e[e["day_index"] < 30]["energyKwh"].sum()
last = e[e["day_index"] >= 30]["energyKwh"].sum()
energy_high = bool(last > prev)
```

This ensures that only contextually relevant and available challenges remain for scoring.

### 2) Model Scoring (Completion Probability / 1-Hour Completion Probability)

For each remaining candidate, the models compute success probabilities:

Main model: probability of completing a daily or monthly challenge today

Speed model: probability of completing a speed challenge within one hour for each possible notification time
```python
scores = main_model.predict_proba(df_main)[:, 1]      # completion probability
probs  = speed_model.predict_proba(df_feat)[:, 1]     # 1-hour completion probability
```

Speed-mode scoring evaluates multiple candidate notification times (06:00–22:00) and selects the most promising time window.

### 3) Score Adjustment: Recommendation Penalty (α × freq) + Softmax Sampling

To prevent repetitive recommendations, each challenge’s score is adjusted using a frequency penalty:
```python
cand["freq"] = cand["challengeId"].map(
    lambda cid: recommend_count.get(cid, 0)
)
cand["adj_score"] = cand["score"] - ALPHA * cand["freq"]
```

freq: how many times the challenge has been recommended

ALPHA: a small coefficient that slightly reduces the score of frequently shown tasks

After adjustment, only the top-K candidates are retained, and Softmax-like sampling introduces diversity:
```python
w = np.exp(cand_top["adj_score"])
p = w / w.sum()
chosen = cand_top.sample(n=1, weights=p).iloc[0]
```

This ensures that high-scoring challenges are preferred, but not always repeated, creating a more engaging user experience.

### 4) Final Recommendation Generation

Finally, the system outputs exactly one challenge per category:

one daily challenge

one monthly challenge (if the energy condition allows)

one speed challenge with the optimized notification time selected by the speed model
```python
daily   = recommend_non_speed("daily")
monthly = recommend_non_speed("monthly")
speed   = recommend_speed()

return {
    "userId": TARGET_USER,
    "daily": daily,
    "monthly": monthly,
    "speed": speed,
    "energyHigh": energy_high,
    "main_auc": auc_main,
    "speed_auc": auc_sp,
}
```

## IV. Evaluation & Analysis
This section presents the evaluation of the Gradient Boosting Decision Tree model trained on six months of simulated HomeQuest activity logs. The goal is not perfect binary prediction but estimating relative completion likelihoods for ranking daily challenges.

A. Dataset and Evaluation Setup

- Input features include weekday, personal points, family points, energy usage, mode, category, duration type, progress type, device type, and user identifier.
- The target output is the completion of each challenge event.
- Data was split chronologically to reflect real-world usage.
- First two-thirds of the timeline used for training.
- Remaining one-third used for testing.

This ensures that the model learns from past events and predicts future user behavior.

B. Main Model Quantitative Results

The main GBDT model produced the following scores on the test set:

- AUC: 0.5587
- RMSE: 0.547
- MAE: 0.4758
- Accuracy: 0.5574

Interpretation of these values:

- User behavior in the dataset fluctuates heavily between zero and one.
- Many important behavioral factors are not captured by available features.
- Perfect classification is unrealistic in this setting.
- The model is intended to generate relative probability scores for ranking challenges, not to make perfect binary predictions.

Therefore, these values are acceptable within the context of recommendation logic.

C. Trend Comparison Using Moving Average

To provide a clearer comparison between predicted probabilities and actual completion behavior, a moving average with a window size of ten samples was applied.
<img width="3000" height="1200" alt="gbdt_smoothed_trend" src="https://github.com/user-attachments/assets/5a14165c-6737-4df8-bf44-c4129212395a" />


Interpretation:

- The smoothed predicted probabilities rise during periods when the smoothed actual completion rate increases.
- Both curves decrease during intervals of lower completion behavior.
- The two curves share similar global patterns even though they do not match event by event.
- This demonstrates that the model captures general behavioral tendencies despite high noise.
- For a recommendation system, capturing trend-level information is more important than exact classification accuracy.

D. Speed Challenge Model

A separate GBDT model was trained to predict whether a speed-type challenge would be completed within one hour after notification. The model reached an AUC of 0.9994 in the simulated environment.

Interpretation:

- The extremely high value is due to the deterministic structure of simulated duration data.
- Training and testing were performed on the same dataset, creating an optimistic evaluation.
- Real-world evaluation should include a proper time-based split.

E. Overall Interpretation

- The main model functions as a ranking model rather than a strict classifier.
- The moving average comparison confirms that the model captures meaningful behavioral patterns.
- The model provides stable signals that are suitable for prioritizing challenge recommendations.
- While the simulation contains noise and limited behavioral variables, the GBDT framework remains a practical foundation for future improvements.

## V. Related Work (e.g., existing studies)

### - **Duan, T., Avati, A., Ding, D., Thai, K., Basu, S., Ng, A., & Schuler, A. (2020). *NGBoost: Natural Gradient Boosting for Probabilistic Prediction.***

Duan et al. (2020) present NGBoost, a probabilistic extension of gradient boosting that overcomes the traditional limitation of producing only point estimates. Instead of predicting a single scalar output, NGBoost models the full conditional distribution by learning multiple parameters—such as mean and variance—through a set of differentiable scoring rules. One of its core contributions is the use of natural gradients, which stabilize the optimization landscape when fitting multiple distribution parameters simultaneously. The authors show that this approach yields well-calibrated uncertainty estimates and robust predictive performance on diverse tabular datasets.

This work is highly relevant to HomeQuest because our system requires not only a binary prediction of whether a challenge will succeed but a well-behaved probability estimate that reflects the confidence of the model under behavioral uncertainty. Household activity data, much like user-generated behavioral logs, tends to be noisy, irregular, and context-dependent. NGBoost demonstrates that gradient-boosting-based models remain expressive and stable under such conditions and reinforces the idea that boosted decision trees can be extended to probabilistic modeling without sacrificing interpretability. While HomeQuest does not implement the NGBoost architecture directly, the paper provides theoretical justification for using GBDT-style models as probability estimators and using those probabilities as the driving signal for stochastic recommendation mechanisms such as softmax sampling.

### - **Friedman, J. H. (2001). *Greedy Function Approximation: A Gradient Boosting Machine.***

Friedman (2001) formulates gradient boosting as a general framework for function approximation by viewing supervised learning as numerical optimization in function space. The paper shows that an additive model composed of many simple base learners—typically shallow regression trees—can be trained by performing stagewise gradient descent on an arbitrary differentiable loss function. In practice, this leads to the now-standard Gradient Boosting Decision Tree (GBDT) procedure, where each tree is fitted to the negative gradient (pseudo-residuals) of the loss, and regularization is controlled through the number of trees, their depth, and the learning rate.

The paper also provides concrete algorithms for several loss functions, including least-squares, least absolute deviation, Huber loss, and logistic loss, and demonstrates that tree-based gradient boosting yields competitive, robust models on noisy tabular data. This work is the classical theoretical foundation for the GBDT model used in our HomeQuest system: we follow this paradigm by training a tree-based gradient boosting classifier on historical challenge logs to approximate the probability that a given challenge will be successfully completed, and then use these predicted probabilities as the primary signal for ranking and recommending daily, monthly, and speed challenges to each user.

## VI. Conclusion: Discussion
