# prompt to obtain user preference summary from sample questions and responses
PAG_SUMMARY_PROMPT = """
Write a summary of {user}'s preferences for responses given their ratings for the sample questions and responses. {direction}

{formatted_samples}
""".strip()

# review length direction
PAG_REVIEW_LENGTH_DIRECTION = (
    "Consider the length of the response when writing your summary."
)

# review level direction
PAG_REVIEW_LEVEL_DIRECTION = (
    "Consider the reading level of the response when writing your summary."
)

# math solutions direction
PAG_MATH_SOLUTIONS_DIRECTION = "Consider whether they like seeing multiple solution methods, asking questions of the student during the solution, providing very detailed or big-picture explanations, and/or using visual guides."

# prompt to generate response to question based on user preferences and sample responses and ratings
PAG_GENERATION_PROMPT = """
Generate a response to {user} for the given question based on {user}'s preferences and sample responses and ratings.

#### START QUESTION ####
{question}
#### END QUESTION ####

#### START USER PREFERENCES ####
{user_preferences}
## END USER PREFERENCES ##

{formatted_samples}
""".strip()

# template for a single sample question and multiple responses and ratings
SAMPLE_TEMPLATE = """
#### START SAMPLE {idx} ####

## START QUESTION ##
{question}
## END QUESTION ##

{formatted_responses_and_ratings}

#### END SAMPLE {idx} ####
""".strip()

# template for a single response and rating
RESPONSE_AND_RATING_TEMPLATE = """
## START RESPONSE {idx} ##
{response}
## END RESPONSE {idx} ##

## START USER RATING {idx} ##
{user}'s rating: {rating} out of 1.0
## END USER RATING {idx} ##
""".strip()
