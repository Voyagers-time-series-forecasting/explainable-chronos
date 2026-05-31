"""
Extension 2 — Query datasets.

EXAMPLE_SET (70 queries): the labeled pool used as few-shot examples by
the BERT tier. It merges what were previously called DEV and TEST sets.

TEST_SET (10 queries): the held-out evaluation set, written after all
patterns and the BERT pool were frozen. This is the only set used for
the reported score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


# Covariate names shared across all evaluation scenarios.
COVARIATE_NAMES: List[str] = [
    "marketing_spend", "website_traffic", "previous_day_sales",
    "competitor_promotion_index", "price_discount_percentage",
    "holiday_proximity", "shipping_delay_hours",
    "social_media_mentions", "weather_temperature", "random_sensor_noise",
]


@dataclass
class TestCase:
    """A single test case for the dialogue evaluation."""
    query: str
    expected_intent: str
    description: str
    expected_covariate: Optional[str] = None
    expected_horizon: Optional[int] = None


@dataclass
class MultiTurnTestCase:
    """A sequence of dialogue turns testing cross-turn state persistence."""
    description: str
    turns: List[TestCase]


# ── Example set (70 queries) ──────────────────────────────────────────
# Few-shot pool for the BERT tier. Combines what were previously the DEV
# (40 queries) and TEST (30 queries) sets, both of which were used during
# rule-based parser development and are therefore not independent test sets.
EXAMPLE_SET: List[TestCase] = [
    # remove_covariate (10)
    TestCase("What would happen if we removed the marketing spend covariate?", "remove_covariate", "Remove marketing_spend by name", expected_covariate="marketing_spend"),
    TestCase("What if there were no website traffic data?", "remove_covariate", "Remove website_traffic with no-data phrasing", expected_covariate="website_traffic"),
    TestCase("Show me the forecast without the shipping delay information.", "remove_covariate", "Remove shipping_delay_hours with 'without' phrasing", expected_covariate="shipping_delay_hours"),
    TestCase("Exclude the competitor promotion index from the model.", "remove_covariate", "Remove competitor_promotion_index with 'exclude' phrasing", expected_covariate="competitor_promotion_index"),
    TestCase("Drop the previous day sales covariate.", "remove_covariate", "Remove previous_day_sales with 'drop' phrasing", expected_covariate="previous_day_sales"),
    TestCase("What if holiday proximity didn't affect the forecast?", "remove_covariate", "Remove holiday_proximity with hypothetical phrasing", expected_covariate="holiday_proximity"),
    TestCase("Zero out the social media mentions variable.", "remove_covariate", "Remove social_media_mentions with 'zero out' phrasing", expected_covariate="social_media_mentions"),
    TestCase("Eliminate weather temperature from the analysis.", "remove_covariate", "Remove weather_temperature with 'eliminate' phrasing", expected_covariate="weather_temperature"),
    TestCase("What would the forecast look like without price discounts?", "remove_covariate", "Remove price_discount_percentage", expected_covariate="price_discount_percentage"),
    TestCase("Remove all covariate effects and show univariate forecast.", "remove_covariate", "Generic remove request without specific covariate"),
    # scale_covariate (10)
    TestCase("What if marketing spend doubled?", "scale_covariate", "Double marketing_spend", expected_covariate="marketing_spend"),
    TestCase("What would happen if website traffic increased by 50%?", "scale_covariate", "Increase website_traffic by 50%", expected_covariate="website_traffic"),
    TestCase("Show me the forecast if price discounts were reduced by 30%.", "scale_covariate", "Reduce price_discount_percentage by 30%", expected_covariate="price_discount_percentage"),
    TestCase("What if social media mentions dropped by 20%?", "scale_covariate", "Decrease social_media_mentions by 20%", expected_covariate="social_media_mentions"),
    TestCase("Halve the shipping delay and show the new forecast.", "scale_covariate", "Halve shipping_delay_hours", expected_covariate="shipping_delay_hours"),
    TestCase("What if competitor promotions tripled in intensity?", "scale_covariate", "Triple competitor_promotion_index", expected_covariate="competitor_promotion_index"),
    TestCase("Scale up marketing spend by 2x.", "scale_covariate", "Scale marketing_spend by 2x", expected_covariate="marketing_spend"),
    TestCase("What if holiday proximity increased by 25%?", "scale_covariate", "Increase holiday_proximity by 25%", expected_covariate="holiday_proximity"),
    TestCase("Reduce website traffic by half.", "scale_covariate", "Halve website_traffic", expected_covariate="website_traffic"),
    TestCase("What would happen if previous day sales rose by 10%?", "scale_covariate", "Increase previous_day_sales by 10%", expected_covariate="previous_day_sales"),
    # change_horizon (10)
    TestCase("Show me the next 7 days instead.", "change_horizon", "Change horizon to 7 days", expected_horizon=168),
    TestCase("Can you forecast the next 30 days?", "change_horizon", "Change horizon to 30 days", expected_horizon=720),
    TestCase("I want to see a 14-day forecast.", "change_horizon", "Change horizon to 14 days", expected_horizon=336),
    TestCase("Predict the next 48 hours.", "change_horizon", "Change horizon to 48 hours", expected_horizon=48),
    TestCase("Show me 10 steps ahead.", "change_horizon", "Change horizon to 10 steps", expected_horizon=10),
    TestCase("Can you extend the forecast to 3 weeks?", "change_horizon", "Change horizon to 3 weeks", expected_horizon=504),
    TestCase("Give me a 24-hour forecast.", "change_horizon", "Change horizon to 24 hours", expected_horizon=24),
    TestCase("Forecast for the next 5 days.", "change_horizon", "Change horizon to 5 days", expected_horizon=120),
    TestCase("Show 50 periods ahead.", "change_horizon", "Change horizon to 50 periods", expected_horizon=50),
    TestCase("I need a 2-week prediction.", "change_horizon", "Change horizon to 2 weeks", expected_horizon=336),
    # confidence_query (10)
    TestCase("How confident are you in this forecast?", "confidence_query", "Direct confidence question"),
    TestCase("What is the uncertainty around your predictions?", "confidence_query", "Uncertainty question"),
    TestCase("What are the best and worst case scenarios?", "confidence_query", "Best/worst case question"),
    TestCase("How wide are the prediction intervals?", "confidence_query", "Prediction interval width question"),
    TestCase("What is the P10 and P90 range?", "confidence_query", "P10/P90 question"),
    TestCase("How certain are you about the next 7 periods?", "confidence_query", "Certainty question"),
    TestCase("What is the downside risk in this forecast?", "confidence_query", "Downside risk question"),
    TestCase("Is there a lot of uncertainty in your prediction?", "confidence_query", "General uncertainty question"),
    TestCase("What is the upside potential according to this model?", "confidence_query", "Upside potential question"),
    TestCase("Can you tell me the margin of error for this forecast?", "confidence_query", "Margin of error question"),
    # remove_covariate (5 more)
    TestCase("Run it again without marketing spend.", "remove_covariate", "Remove marketing_spend with alternate without phrasing", expected_covariate="marketing_spend"),
    TestCase("Remove website traffic from the inputs before forecasting.", "remove_covariate", "Remove website_traffic from inputs", expected_covariate="website_traffic"),
    TestCase("Can you drop shipping delay hours for this run?", "remove_covariate", "Remove shipping_delay_hours with question phrasing", expected_covariate="shipping_delay_hours"),
    TestCase("What if the competitor promotion index were gone?", "remove_covariate", "Remove competitor_promotion_index with gone phrasing", expected_covariate="competitor_promotion_index"),
    TestCase("Set previous day sales to zero and forecast again.", "remove_covariate", "Remove previous_day_sales with zero phrasing", expected_covariate="previous_day_sales"),
    # scale_covariate (5)
    TestCase("Scale marketing spend to 2x and rerun the forecast.", "scale_covariate", "Double marketing_spend with multiplier phrasing", expected_covariate="marketing_spend"),
    TestCase("What changes if website traffic is 40% higher?", "scale_covariate", "Increase website_traffic by 40%", expected_covariate="website_traffic"),
    TestCase("Rerun the forecast with discounts 15% lower.", "scale_covariate", "Reduce price_discount_percentage by 15%", expected_covariate="price_discount_percentage"),
    TestCase("Cut social media mentions by 25%.", "scale_covariate", "Reduce social_media_mentions by 25%", expected_covariate="social_media_mentions"),
    TestCase("Increase weather temperature by 10% and show the result.", "scale_covariate", "Increase weather_temperature by 10%", expected_covariate="weather_temperature"),
    # change_horizon (5)
    TestCase("Forecast the next 72 hours.", "change_horizon", "Change horizon to 72 hours", expected_horizon=72),
    TestCase("Show a 4-day forecast.", "change_horizon", "Change horizon to 4 days", expected_horizon=96),
    TestCase("Extend the forecast to 3 weeks.", "change_horizon", "Change horizon to 3 weeks", expected_horizon=504),
    TestCase("Use a horizon of 12 steps.", "change_horizon", "Change horizon to 12 periods", expected_horizon=12),
    TestCase("Predict 1 month ahead.", "change_horizon", "Change horizon to 1 month", expected_horizon=720),
    # confidence_query (5)
    TestCase("How confident is this projection?", "confidence_query", "Reliability question"),
    TestCase("How uncertain is the forecast over the next few periods?", "confidence_query", "Stability and uncertainty question"),
    TestCase("Do the prediction intervals spread out much?", "confidence_query", "Forecast band width question"),
    TestCase("What range should I expect for the forecast?", "confidence_query", "Risk around prediction question"),
    TestCase("Is there much downside risk in these predictions?", "confidence_query", "Tightness and uncertainty question"),
    # fuzzy covariate names (4)
    TestCase("What happens if we remove the marketing budget?", "remove_covariate", "Remove marketing_spend via alias 'marketing budget'", expected_covariate="marketing_spend"),
    TestCase("What if website visits increased by 30%?", "scale_covariate", "Scale website_traffic via alias 'website visits'", expected_covariate="website_traffic"),
    TestCase("How reliable is this prediction, really?", "confidence_query", "Informal confidence question without keywords"),
    TestCase("Show only the next 2 weeks.", "change_horizon", "Change horizon to 2 weeks via 'only'", expected_horizon=336),
    # counterfactual (2)
    TestCase("What would have happened if marketing spend had been higher last month?", "counterfactual", "Historical counterfactual — system should decline", expected_covariate="marketing_spend"),
    TestCase("What if website traffic had been much higher last week?", "counterfactual", "Historical counterfactual about past data", expected_covariate="website_traffic"),
    # edge cases (4)
    TestCase("Give me a one-month forecast.", "change_horizon", "Change horizon — 'one-month' as word not digit", expected_horizon=720),
    TestCase("Can you drop the social media variable?", "remove_covariate", "Remove via 'drop' + alias 'social media variable'", expected_covariate="social_media_mentions"),
    TestCase("Triple the competitor promotion index.", "scale_covariate", "Scale with word-factor 'triple'", expected_covariate="competitor_promotion_index"),
    TestCase("What are the P10 and P90 bounds?", "confidence_query", "Explicit quantile names"),

    # ── additional examples (30) ─────────────────────────────────────
    # remove_covariate (6)
    TestCase("What happens if we ignore the social media data?", "remove_covariate", "Remove social_media_mentions via 'ignore'", expected_covariate="social_media_mentions"),
    TestCase("Omit the holiday effect from the model.", "remove_covariate", "Remove holiday_proximity via 'omit'", expected_covariate="holiday_proximity"),
    TestCase("Take price discounts out of the equation.", "remove_covariate", "Remove price_discount_percentage with idiomatic phrasing", expected_covariate="price_discount_percentage"),
    TestCase("Forecast without accounting for weather.", "remove_covariate", "Remove weather_temperature with 'without accounting for'", expected_covariate="weather_temperature"),
    TestCase("Suppress the competitor promotion variable.", "remove_covariate", "Remove competitor_promotion_index via 'suppress'", expected_covariate="competitor_promotion_index"),
    TestCase("What if random noise weren't included?", "remove_covariate", "Remove random_sensor_noise with hypothetical phrasing", expected_covariate="random_sensor_noise"),
    # scale_covariate (6)
    TestCase("Increase marketing spend by 40%.", "scale_covariate", "Scale marketing_spend up by 40%", expected_covariate="marketing_spend"),
    TestCase("What if shipping delays were cut by half?", "scale_covariate", "Scale shipping_delay_hours to 0.5x", expected_covariate="shipping_delay_hours"),
    TestCase("Grow social media mentions by 60%.", "scale_covariate", "Scale social_media_mentions up by 60%", expected_covariate="social_media_mentions"),
    TestCase("What if website traffic dropped 10%?", "scale_covariate", "Scale website_traffic down by 10%", expected_covariate="website_traffic"),
    TestCase("Apply a 20% reduction to the competitor index.", "scale_covariate", "Scale competitor_promotion_index down by 20%", expected_covariate="competitor_promotion_index"),
    TestCase("Boost previous day sales by 15%.", "scale_covariate", "Scale previous_day_sales up by 15%", expected_covariate="previous_day_sales"),
    # change_horizon (6)
    TestCase("Show me predictions for the next 100 hours.", "change_horizon", "Change horizon to 100 hours", expected_horizon=100),
    TestCase("I need a 3-day forecast.", "change_horizon", "Change horizon to 3 days", expected_horizon=72),
    TestCase("What's the outlook for the next 6 months?", "change_horizon", "Change horizon to 6 months", expected_horizon=4320),
    TestCase("Run a 500-step forecast.", "change_horizon", "Change horizon to 500 steps", expected_horizon=500),
    TestCase("Give me 1 week ahead.", "change_horizon", "Change horizon to 1 week", expected_horizon=168),
    TestCase("Forecast for the next 2 weeks.", "change_horizon", "Change horizon to 2 weeks", expected_horizon=336),
    # confidence_query (6)
    TestCase("What's the error margin on this?", "confidence_query", "Confidence via error margin phrasing"),
    TestCase("How spread out are the predictions?", "confidence_query", "Confidence via spread phrasing"),
    TestCase("Tell me the optimistic and pessimistic scenarios.", "confidence_query", "Confidence via scenario phrasing"),
    TestCase("Are the prediction bounds tightening or widening?", "confidence_query", "Confidence via bounds trend"),
    TestCase("What level of confidence should I assign to this?", "confidence_query", "Confidence as an assignment question"),
    TestCase("How much should I rely on this forecast?", "confidence_query", "Confidence via reliability phrasing"),
    # counterfactual (6)
    TestCase("What if marketing spend had been higher last month?", "counterfactual", "Historical counterfactual on marketing_spend", expected_covariate="marketing_spend"),
    TestCase("Suppose sales last week had been 30% lower.", "counterfactual", "Historical counterfactual on previous_day_sales", expected_covariate="previous_day_sales"),
    TestCase("What would the result have been with double the website traffic last quarter?", "counterfactual", "Historical counterfactual on website_traffic", expected_covariate="website_traffic"),
    TestCase("If price discounts were halved a month ago, how would that have changed things?", "counterfactual", "Historical counterfactual on price_discount_percentage", expected_covariate="price_discount_percentage"),
    TestCase("What if there had been no social media activity last week?", "counterfactual", "Historical counterfactual on social_media_mentions", expected_covariate="social_media_mentions"),
    TestCase("How would higher shipping delays last month have changed the outcome?", "counterfactual", "Historical counterfactual on shipping_delay_hours", expected_covariate="shipping_delay_hours"),
]

# ── Test set (40 queries) ─────────────────────────────────────────────
# Written after all patterns and the BERT pool were frozen.
# Never seen during development. This is the reported evaluation set.
# Breakdown: 10 original held-out + 16 ambiguous + 14 outlier.
TEST_SET: List[TestCase] = [
    # ── original held-out cases (10) ─────────────────────────────────
    TestCase("How much uncertainty surrounds this prediction?", "confidence_query", "Uncertainty question without the word 'confidence'"),
    TestCase("Can you tell me the prediction intervals?", "confidence_query", "Direct prediction-interval request"),
    TestCase("Exclude weather_temperature from the model.", "remove_covariate", "Remove via 'exclude' with exact name", expected_covariate="weather_temperature"),
    TestCase("What if there were no holiday_proximity factor?", "remove_covariate", "Remove via 'what if...no' phrasing", expected_covariate="holiday_proximity"),
    TestCase("Double the holiday_proximity.", "scale_covariate", "Scale with word-factor 'double'", expected_covariate="holiday_proximity"),
    TestCase("Suppose social_media_mentions dropped by 40%.", "scale_covariate", "Scale via 'dropped by' + percentage", expected_covariate="social_media_mentions"),
    TestCase("What if the price discount were halved?", "scale_covariate", "Scale via 'halved' + partial alias 'price discount'", expected_covariate="price_discount_percentage"),
    TestCase("Forecast for the next three days.", "change_horizon", "Horizon via word-number 'three days'", expected_horizon=72),
    TestCase("Can we look at 96 steps ahead?", "change_horizon", "Horizon via 'N steps ahead' phrasing", expected_horizon=96),
    TestCase("What would the forecast have looked like with higher marketing spend last quarter?", "counterfactual", "Historical counterfactual — no scale trigger word", expected_covariate="marketing_spend"),

    # ── ambiguous phrasing (8) ────────────────────────────────────────
    # Queries where phrasing could suggest the wrong intent.
    TestCase("Pretend holiday_proximity didn't exist.", "remove_covariate", "Ambiguous: 'pretend...didn't exist' looks counterfactual but is forward-looking", expected_covariate="holiday_proximity"),
    TestCase("What if marketing effectiveness were halved?", "scale_covariate", "Ambiguous: 'effectiveness' is an alias, 'were' looks counterfactual", expected_covariate="marketing_spend"),
    TestCase("Let's see what happens without any sensor noise.", "remove_covariate", "Ambiguous: indirect alias 'sensor noise' for random_sensor_noise", expected_covariate="random_sensor_noise"),
    TestCase("Suppose traffic on the website soared by 50%.", "scale_covariate", "Ambiguous: 'soared' is a scale marker but 'suppose' could suggest counterfactual", expected_covariate="website_traffic"),
    TestCase("Show me a scenario where social media impact doubled.", "scale_covariate", "Ambiguous: 'scenario' + past tense 'doubled' could be counterfactual", expected_covariate="social_media_mentions"),
    TestCase("Could you extend the view by 2 weeks?", "change_horizon", "Ambiguous: 'extend the view' is an indirect way to say change horizon", expected_horizon=336),
    TestCase("What if sales yesterday had been 20% higher?", "counterfactual", "Ambiguous: 'yesterday' makes this historical despite scale phrasing", expected_covariate="previous_day_sales"),
    TestCase("Is there a chance this is just noise?", "confidence_query", "Ambiguous: philosophical phrasing, no explicit confidence keyword"),

    # ── outlier phrasings (7) ─────────────────────────────────────────
    # Unusual or indirect phrasings rules will not match; BERT or LLM needed.
    TestCase("Try ignoring the competitor data entirely.", "remove_covariate", "Outlier: 'competitor data' as alias, 'try ignoring' as remove verb", expected_covariate="competitor_promotion_index"),
    TestCase("Is this model trustworthy for real business decisions?", "confidence_query", "Outlier: trust/reliability framing with no standard keywords"),
    TestCase("Give me a longer outlook — say 5 days.", "change_horizon", "Outlier: informal 'longer outlook' with clarifying aside", expected_horizon=120),
    TestCase("What does the picture look like with shipping times cut in half?", "scale_covariate", "Outlier: very indirect phrasing, 'picture' as forecast synonym", expected_covariate="shipping_delay_hours"),
    TestCase("I want to stress-test the effect of discounts — boost them by 30%.", "scale_covariate", "Outlier: analytical framing with 'stress-test', factor embedded mid-sentence", expected_covariate="price_discount_percentage"),
    TestCase("Honestly, how confident should I be in these numbers?", "confidence_query", "Outlier: conversational register, no forecast-domain keywords"),
    TestCase("Project forward 48 hours from now.", "change_horizon", "Outlier: 'project forward' as forecast verb with time-from-now phrasing", expected_horizon=48),

    # ── additional ambiguous cases (8) ────────────────────────────────
    TestCase("What if we paused all advertising?", "remove_covariate", "Ambiguous: 'paused' is an unusual remove verb, could look like scale", expected_covariate="marketing_spend"),
    TestCase("Turn off the weather data for now.", "remove_covariate", "Ambiguous: 'turn off' is informal, could trigger scale path", expected_covariate="weather_temperature"),
    TestCase("Bump website traffic up by 25%.", "scale_covariate", "Ambiguous: 'bump up' is slang, 'by a quarter' is an indirect factor", expected_covariate="website_traffic"),
    TestCase("How would things look over the next 2 months?", "change_horizon", "Ambiguous: 'how would things look' could parse as confidence_query", expected_horizon=1440),
    TestCase("What does the model think about uncertainty in its outputs?", "confidence_query", "Ambiguous: meta question about the model could misfire as remove or unknown"),
    TestCase("Assume price discounts simply vanished.", "remove_covariate", "Ambiguous: 'vanished' reads like a counterfactual past-tense but is forward-looking", expected_covariate="price_discount_percentage"),
    TestCase("Previous day sales were 50% stronger — what would have changed?", "counterfactual", "Ambiguous: 'were' + scale phrasing, could trigger scale_covariate", expected_covariate="previous_day_sales"),
    TestCase("Bring previous day sales up to 1.5x its current level.", "scale_covariate", "Ambiguous: 'bring up to X of current' is an indirect multiplier phrasing", expected_covariate="previous_day_sales"),

    # ── additional outlier phrasings (7) ──────────────────────────────
    TestCase("Kill the shipping delays.", "remove_covariate", "Outlier: aggressive informal verb 'kill', no remove keyword in patterns", expected_covariate="shipping_delay_hours"),
    TestCase("Is this prediction solid enough to act on?", "confidence_query", "Outlier: action-oriented framing, no uncertainty or interval keywords"),
    TestCase("Give me a 10-day window.", "change_horizon", "Outlier: 'window' instead of forecast/horizon, digit + day unit", expected_horizon=240),
    TestCase("Neutralize the holiday proximity effect.", "remove_covariate", "Outlier: 'neutralize' as remove verb, not in any pattern catalog", expected_covariate="holiday_proximity"),
    TestCase("How stable is the forecast under current conditions?", "confidence_query", "Outlier: stability framing, no P10/P90 or confidence keyword"),
    TestCase("Competitor index up 2x — run it.", "scale_covariate", "Outlier: telegraphic style, no full sentence structure", expected_covariate="competitor_promotion_index"),
    TestCase("Peek at the next 6 hours.", "change_horizon", "Outlier: 'peek' as forecast verb, short horizon edge case", expected_horizon=6),

    # ── additional standard cases (7) ────────────────────────────────
    TestCase("Remove the marketing covariate.", "remove_covariate", "Standard: direct remove with intent keyword", expected_covariate="marketing_spend"),
    TestCase("Increase website traffic by 35%.", "scale_covariate", "Standard: clear percentage scale", expected_covariate="website_traffic"),
    TestCase("I want a 20-day forecast.", "change_horizon", "Standard: digit + day unit", expected_horizon=480),
    TestCase("What's the P90 for this forecast?", "confidence_query", "Standard: direct quantile request"),
    TestCase("Show me 36 hours ahead.", "change_horizon", "Standard: digit + hour unit", expected_horizon=36),
    TestCase("Multiply the shipping delays by 1.5.", "scale_covariate", "Standard: explicit multiplier phrasing", expected_covariate="shipping_delay_hours"),
    TestCase("What if competitor promotions hadn't happened last quarter?", "counterfactual", "Standard: historical counterfactual with clear past-tense marker", expected_covariate="competitor_promotion_index"),

    # ── additional ambiguous cases (7) ───────────────────────────────
    TestCase("What happens if we ditch the weather input?", "remove_covariate", "Ambiguous: 'ditch' is informal, 'input' indirect alias", expected_covariate="weather_temperature"),
    TestCase("Slash shipping delays by 40%.", "scale_covariate", "Ambiguous: 'slash' implies decrease but is not a standard scale keyword", expected_covariate="shipping_delay_hours"),
    TestCase("Look ahead 4 weeks.", "change_horizon", "Ambiguous: 'look ahead' could parse as confidence_query", expected_horizon=672),
    TestCase("Am I right to trust this model?", "confidence_query", "Ambiguous: trust framing, no forecast-domain keywords"),
    TestCase("What if social media influence had been negligible?", "counterfactual", "Ambiguous: 'had been negligible' could look like remove_covariate", expected_covariate="social_media_mentions"),
    TestCase("Scale the noise factor down by 50%.", "scale_covariate", "Ambiguous: 'noise factor' as alias for random_sensor_noise", expected_covariate="random_sensor_noise"),
    TestCase("What if price cuts had been more aggressive last year?", "counterfactual", "Ambiguous: 'price cuts' alias + 'had been' past tense + scale trigger", expected_covariate="price_discount_percentage"),

    # ── additional outlier cases (6) ─────────────────────────────────
    TestCase("Mute the holiday signal.", "remove_covariate", "Outlier: 'mute' as remove verb, no pattern match", expected_covariate="holiday_proximity"),
    TestCase("How much wiggle room is there in these numbers?", "confidence_query", "Outlier: very informal, no forecast vocabulary"),
    TestCase("Stretch the forecast out to 3 months.", "change_horizon", "Outlier: 'stretch out' as horizon verb", expected_horizon=2160),
    TestCase("What would a world without competitor promotions look like?", "remove_covariate", "Outlier: rhetorical framing, very indirect", expected_covariate="competitor_promotion_index"),
    TestCase("Fire up the marketing budget by 80%.", "scale_covariate", "Outlier: 'fire up' slang + 'marketing budget' alias", expected_covariate="marketing_spend"),
    TestCase("How sure are we about the direction of this forecast?", "confidence_query", "Outlier: 'direction' + 'how sure', non-standard confidence framing"),
]

# ── Multi-turn test set ────────────────────────────────────────────────
MULTI_TURN_SET: List[MultiTurnTestCase] = [
    MultiTurnTestCase(
        description="Remove covariate then query confidence",
        turns=[
            TestCase("Remove marketing_spend from the forecast.", "remove_covariate", "Turn 1: remove covariate", expected_covariate="marketing_spend"),
            TestCase("How confident are you in this forecast?", "confidence_query", "Turn 2: confidence query — state from turn 1 should persist"),
        ],
    ),
    MultiTurnTestCase(
        description="Scale covariate then change horizon",
        turns=[
            TestCase("What if website traffic increased by 50%?", "scale_covariate", "Turn 1: scale covariate", expected_covariate="website_traffic"),
            TestCase("Show me the next 7 days.", "change_horizon", "Turn 2: change horizon — scaled covariates should persist", expected_horizon=168),
        ],
    ),
    MultiTurnTestCase(
        description="Two successive covariate modifications",
        turns=[
            TestCase("Remove shipping delay hours.", "remove_covariate", "Turn 1: remove first covariate", expected_covariate="shipping_delay_hours"),
            TestCase("Also double the marketing spend.", "scale_covariate", "Turn 2: scale second covariate — first should remain zeroed", expected_covariate="marketing_spend"),
        ],
    ),
]
