## 39th Solution - Detect Sleep States

**Thanks to Kaggle for hosting this meaningful competition. **

**Thanks to all of my teammates for gaining the silver medal together.**

Main GitHub Repo: [Here](https://github.com/lullabies777/kaggle-detect-sleep)

PrecTime GitHub Repo: [Here](https://github.com/Lizhecheng02/Kaggle-Detect_Sleep_States)

### Here is the Detail Solution

#### Baseline Code

Here, we need to thank [tubotubo](https://www.kaggle.com/tubotubo) for providing the baseline code.  We didn't join the competition from the very beginning, this baseline code provided us with some ideas and basic model structures.

#### Dataset Preparation

- We didn't use any methods to handle the dirty data, which might be one reason why we couldn't improve our scores anymore.

- On the evening before the competition ended, my teammate found this [discussion](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/456177). Consequently, we attempted to clean the data by removing the data for the days where the event was empty. However, due to the time limitation, we didn't make significant progress.  

- We believe this should be helpful because the model using this method showed a smaller difference in scores on the private leaderboard.

  ![0.739 - Clean Data Model](C:\Users\86183\Desktop\微信图片_20231205195307.png)

#### Feature Engineering

We generated new features using shift, different and rolling window functions. 

The final set we utilized comprised 24 rolling features in addition to the original 4, making a total of 28 features.

Code: [Here](https://github.com/lullabies777/kaggle-detect-sleep/blob/main/run/prepare_data.py)

```
 *[pl.col("anglez").diff(i).alias(f"anglez_diff_{i}") for i in range(diff_start, diff_end, diff_step)],
*[pl.col("enmo").diff(i).alias(f"enmo_diff_{i}") for i in range(diff_start, diff_end, diff_step)],
*[pl.col("anglez").shift(i).alias(f"anglez_lag_{i}")
  for i in range(shift_start, shift_end, shift_step) if i != 0],
*[pl.col("enmo").shift(i).alias(f"enmo_lag_{i}")
  for i in range(shift_start, shift_end, shift_step) if i != 0],
*[pl.col("anglez").rolling_mean(window_size).alias(
    f"anglez_mean_{window_size}") for window_size in window_steps],
*[pl.col("anglez").rolling_min(window_size).alias(
    f"anglez_min_{window_size}") for window_size in window_steps],
*[pl.col("anglez").rolling_max(window_size).alias(
    f"anglez_max_{window_size}") for window_size in window_steps],
*[pl.col("anglez").rolling_std(window_size).alias(
    f"anglez_std_{window_size}") for window_size in window_steps],
*[pl.col("enmo").rolling_mean(window_size).alias(
    f"enmo_mean_{window_size}") for window_size in window_steps],
*[pl.col("enmo").rolling_min(window_size).alias(
    f"enmo_min_{window_size}") for window_size in window_steps],
*[pl.col("enmo").rolling_max(window_size).alias(
    f"enmo_max_{window_size}") for window_size in window_steps],
*[pl.col("enmo").rolling_std(window_size).alias(
    f"enmo_std_{window_size}") for window_size in window_steps],
```

#### Wandb sweep

Wandb sweep is a hyperparameter optimization tool provided by the Wandb machine learning platform. It allows automatic exploration of different hyperparameter combinations to enhance a model's performance.

Implementation: [Here](https://github.com/lullabies777/kaggle-detect-sleep/tree/main/run/sweep)

#### Models

- Used overlap - To enhance accuracy in predicting sequence edges, we utilized overlap by using a 10000 length sequence to predict an 8000 length sequence.
- Implementation of PrecTime Model -  You can find details in this [discussion](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/459616). We also made modifications to its structure, including the addition of transformer architecture and residual connection structures. The experiments had shown that these modifications contribute to a certain improvement in the model's performance.

#### Post-preprocessing Trick

We used dynamic programming algorithm to deal with overlap problem.  

**Principle behind this method: To achieve a high MAP (Mean Average Precision), three main criteria need to be met: Firstly, the predicted label should be sufficiently close to the actual label. Secondly, within a positive or negative tolerance range around the actual label, there should only be one predicted point. Thirdly, the score of other predicted points outside the actual label range should be lower than those within the range.**

```
def get_results_slide_window(pred, gap):
    scores = list(pred)
    stack = [0]
    dp = [-1] * len(scores)
    dp[0] = 0
    for i in range(1,len(scores)):
        if i - stack[-1] < gap:
            if scores[i] >= scores[stack[-1]]:
                stack.pop()
                if i - gap >= 0:
                    if stack:
                        if dp[i - gap] != stack[-1]:
                            while stack and dp[i - gap] - stack[-1] < gap:
                                stack.pop()
                            stack.append(dp[i - gap])
                    else:
                        stack.append(dp[i - gap])
                stack.append(i)
        else:
            stack.append(i)
        dp[i] = stack[-1]
    return stack
```

### Ensemble

Our final ensemble method essentially involved averaging different outputs. With post-processing and this ensemble method combined, our results generally follow the pattern that the more models we use or the greater the variety of models, the higher the score.

Our final submission included 30 models: 4 * 5 folds + 10 single models = 30 models

#### What You Can Learn from This Competition:

- [2nd stage: limit the number of events per day](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/459627)

- [Weighted Box Fusion](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/459637)
- [Wavenet](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/459598), [Unet](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/459637)

- [BiLSTM](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/459604)

