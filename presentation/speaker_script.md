# Monte Carlo Pricing of SPY Options — Slide Content and Speaker Script

This deck has ten slides and is designed for a presentation of about 8–10 minutes. Bam presents Slides 2–3 and 7–8. Minh presents Slides 4–6 and 9–10. Slide 1 can be shared, with one student introducing the title in one sentence.

## Slide 1 — Monte Carlo Pricing of SPY Options

**On the slide**

- Title, course, names, and the guiding question: “Can simulated expected payoffs explain option prices?”

**Speaker notes**

“Our project asks whether Monte Carlo simulation can price stock options accurately. We first check the mathematics against Black–Scholes, then compare the model with actual SPY option prices.”

**Time:** 15 seconds.

## Slide 2 — Options as discounted expected values

**Owner:** Bam

**On the slide**

- Call and put payoff functions.
- The discounted expected-payoff formula.
- Notebook figure showing simulated paths and the distribution of terminal prices.

**Speaker notes**

“An option’s value depends on the stock price at expiry, which we do not know today. A call pays when the stock ends above the strike, while a put pays when it ends below the strike. Because the terminal price is random, the fair price is the expected payoff discounted back to today.”

**Time:** 50 seconds.

## Slide 3 — Research questions

**Owner:** Bam

**On the slide**

- Mathematical validation against Black–Scholes.
- Market comparison using SPY data.

**Speaker notes**

“We separate two questions. First, can Monte Carlo reproduce the correct price under the model? Second, even if it does, can the model match prices paid in the real market? These are different tests: the first tests our computation; the second tests the assumptions.”

**Handoff to Minh**

“To answer either question, we first need a way to generate possible future stock prices.”

**Time:** 35 seconds.

## Slide 4 — Stock-price model: geometric Brownian motion

**Owner:** Minh

**On the slide**

- Geometric Brownian motion terminal-price formula.
- Definitions for $S_0$, $r$, $\sigma$, $T$, and $Z$.
- Notebook plot of simulated paths and terminal prices.

**Speaker notes**

“We assume the stock follows geometric Brownian motion. The only random part is Z, a standard normal draw. The equation combines this random shock with a drift determined by the risk-free rate and volatility. It produces positive prices and a right-skewed lognormal distribution, which is a convenient basic model for stock prices.”

**Time:** 55 seconds.

## Slide 5 — Monte Carlo estimator and confidence interval

**Owner:** Minh

**On the slide**

- Monte Carlo pricing estimator.
- Standard-error and confidence-interval explanation.
- Short excerpt of the actual simulation code.

**Speaker notes**

“For every simulated terminal price, we calculate the option payoff and discount it. The average of those discounted payoffs is our estimated option price. The Law of Large Numbers says this average approaches the true expected value as the number of simulations rises. The Central Limit Theorem gives a standard error, so we can report a 95% confidence interval rather than just one random estimate. The code on the right is the core of the implementation used in the notebook.”

**Time:** 60 seconds.

## Slide 6 — Real-market study design

**Owner:** Minh

**On the slide**

- 4 November 2026 SPY snapshot.
- Values for $S_0$, $r$, historical $\sigma$, simulation count, and final sample.
- Notebook figure of 30-day and 60-day realised volatility.

**Speaker notes**

“For the real-market test, we use a SPY option snapshot from 4 November 2026. The stock price was $754.35, the T-bill rate was 3.95%, and we estimated annual volatility as 10% from the previous 60 trading days. We used 200,000 simulations per expiry. To make the comparison more meaningful, we removed far-from-the-money, low-priced, stale, and zero-volume options. This left 389 contracts.”

**Handoff to Bam**

“Before interpreting the real-market result, we need to check that the simulation calculates the model price correctly.”

**Time:** 60 seconds.

## Slide 7 — Validation against Black–Scholes

**Owner:** Bam

**On the slide**

- Call and put Monte Carlo prices.
- Their 95% confidence intervals.
- Black–Scholes values.

**Speaker notes**

“In the test case, both exact Black–Scholes prices fall inside the Monte Carlo confidence intervals. Therefore, the small difference between the simulation and the formula is consistent with random sampling variation. This validates the pricing implementation under the constant-volatility model.”

**Time:** 45 seconds.

## Slide 8 — Monte Carlo error shrinks at the CLT rate

**Owner:** Bam

**On the slide**

- Repeated-estimate convergence chart from the notebook.
- Observed and theoretical error rates.

**Speaker notes**

“We repeated the simulation 500 times at several simulation sizes. The observed spread follows the theoretical one-over-root-N relationship. This is why increasing the number of simulations helps, but with diminishing returns: four times as many simulations are needed to halve the standard error.”

**Handoff to Minh**

“Now that the method has passed both checks, we can compare it with real SPY option prices.”

**Time:** 45 seconds.

## Slide 9 — Market comparison: historical volatility prices options too low

**Owner:** Minh

**On the slide**

- Percentage-error chart from the market-comparison notebook.
- Results on the near-the-money gap and Monte Carlo precision.

**Speaker notes**

“For almost every retained option, our price is below the market price. Near the money, the model is lower by around 40%, or about seven dollars. This is not a failure of Monte Carlo: the Monte Carlo and Black–Scholes prices are within 12 cents, and the typical simulation margin of error is only about five cents. The much larger gap must come from the modelling inputs.”

**Time:** 55 seconds.

## Slide 10 — Volatility skew and conclusion

**Owner:** Minh

**On the slide**

- Implied-volatility skew chart from the notebook.
- Historical versus implied volatility values.
- Short conclusion and limitations.

**Speaker notes**

“The key input is volatility. The previous 60 days suggested only 10% volatility, but market prices imply around 17% near the money. Low-strike puts imply almost 25%, because investors are willing to pay more for protection against a sharp fall. This downward slope across strikes is the volatility skew. Our conclusion is that Monte Carlo works and behaves as the LLN and CLT predict, but a flat, historical volatility assumption does not capture the future uncertainty and crash risk priced by the market. We should also remember that SPY pays dividends, its options allow early exercise, and our snapshot includes some post-market transaction prices.”

**Time:** 70 seconds.

## Transitions and timing summary

| Segment | Slides | Presenter | Target time |
|---|---:|---|---:|
| Opening and motivation | 1–3 | Shared / Bam | 1.5 minutes |
| Methodology | 4–6 | Minh | 3 minutes |
| Numerical validation | 7–8 | Bam | 1.5 minutes |
| Market results and conclusion | 9–10 | Minh | 2 minutes |
| **Total** | **1–10** |  | **about 8 minutes** |

## Presentation reminders

- Use “the market prices in more volatility” rather than claiming an option is overvalued.
- Point to the confidence intervals when discussing uncertainty; do not read every number in the tables.
- On Slide 10, make the distinction explicit: the simulation method works, while the constant-volatility input fails to match the market.
