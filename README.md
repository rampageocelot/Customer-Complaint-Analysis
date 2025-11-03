
# # Consumer Complaint Sentiment Analysis
#
# ## Project Objective
#
# This project aims to analyze consumer complaint narratives to understand common issues and build a sentiment analysis model to classify the sentiment of these complaints on a 1-5 scale.
#
# ## Key Steps
#
# *   **Data Loading:** The consumer complaints data was loaded into a pandas DataFrame, handling potential encoding issues.
# *   **Data Cleaning:** Missing values in several columns were identified and addressed by dropping columns with excessive missing data ('Consumer disputed?', 'Tags') and imputing missing values in 'State' and 'Sub-issue' with the mode. Punctuation, stopwords, and multiple 'X' characters were removed from the consumer complaint narratives, and the text was stemmed to get the root words.
# *   **Exploratory Data Analysis (EDA):** The most common root words in the cleaned text were identified. The top 10 most frequent products, sub-products, and issues were determined and visualized using bar charts.
# *   **Sentiment Scoring (VADER):** The VADER sentiment analysis tool was used to assign a compound sentiment score to each cleaned narrative. This score was then mapped to a 1-5 sentiment rating scale.
# *   **Model Building (Random Forest):** The stemmed narratives were transformed into TF-IDF features. A Random Forest classifier was trained on this data to predict the sentiment rating. The model's performance was evaluated using accuracy, precision, recall, and F1-score.
# *   **OpenAI Integration (Attempted):** An attempt was made to use the OpenAI API with GPT-3.5 for further analysis, though this step was limited due to OpenAI credit availability.
#
# ## Key Findings
#
# *   **Most Frequent Complaint Topics:** Based on the analysis, the most common complaint topics are related to "Credit reporting, credit repair services, or other personal consumer reports" at the product level, "Credit reporting" at the sub-product level, and "Incorrect information on your report" at the issue level.
# *   **Sentiment Rating Distribution:** The VADER sentiment analysis resulted in the following distribution of sentiment ratings on a 1-5 scale:
#     *   Rating 1: 10777 complaints (Very Negative)
#     *   Rating 2: 5771 complaints (Negative)
#     *   Rating 3: 3195 complaints (Neutral)
#     *   Rating 4: 7921 complaints (Positive)
#     *   Rating 5: 15095 complaints (Very Positive)
#
# ## Model Description and Performance
#
# A Random Forest Classifier was used to predict the sentiment rating (on a 1-5 scale) based on the TF-IDF features extracted from the stemmed consumer complaint narratives. Random Forest is an ensemble learning method that constructs a multitude of decision trees at training time and outputs the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.
#
# The model's performance on the test set was evaluated using the following metrics:
#
# *   **Accuracy:** 0.8275
# *   **Precision (weighted):** 0.8349
# *   **Recall (weighted):** 0.8275
# *   **F1-score (weighted):** 0.8245
#
# These metrics indicate that the model performs reasonably well in predicting the sentiment rating of consumer complaints.
#
# ## OpenAI Integration (with caveat)
#
# An attempt was made to leverage the OpenAI API, specifically the `gpt-3.5-turbo` model, to gain deeper insights from the consumer complaint narratives. The intended uses were:
#
# 1.  **Summarizing Sample Narratives:** To generate concise 1-2 sentence summaries of a random sample of complaint narratives to quickly grasp the core issues.
# 2.  **Analyzing Low Sentiment Complaints:** To prompt the model to explain the reasons behind low sentiment scores in selected narratives or provide constructive feedback.
# 3.  **Identifying Predictive Words/Themes:** To analyze cleaned text grouped by sentiment rating and ask GPT-3.5 to identify words and themes most predictive of each rating.
#
