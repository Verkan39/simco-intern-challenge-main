TEST 1:

OBJECTIVE;
knowing whether the three parameters, price, channel and shelf-message are independent? 

TEST:
I will take a particular label(s), example, entrant_price=3.49, or entrant_price=4.49, channel=online, something like this.
Now for this label(s), I see when these labels were given how many of them converted to sales. Now the question is wont we miss the repeat orders?
So to cater the repeat problem, we can tweak this algo a little. What I do is I count the number of times entrant was chosen by that household after the first time the given label(s) convert it to sales.
Then I check independence with classical probability rules for pairs and then in trios. [ P(A∩B) = P(A)·P(B) ]  

EXPECTATION:
I expect to see P(A∩B) = P(A)·P(B) the equality holding nearly true. If that happens I am in a state of doubt whether this is actual relation or is this another noise considered as relation. If the equality isnt holding true, I am sure they are independent.

RAN experiments/label_probabilities.py

RESULT & CONCLUSION :
The result is stored in rest/result_test1.txt.
So my idea failed at this point, not completely, but partially. we computed values of p(A), p(A and B), p(A and B and C) across all available cominations, and we can check independence of combinations easily, but rather that checking independence of combinations, I am more concerened about independence of "price", "channel" and "entrant_message". Now I first though of using p-values with Chi-Square, but that doesnt work because for that we need to assume two rows to be independent, which they clearly are not. 


TEST 2:

OBJECTIVE:
understanding how each value of price, channel and shelf-message change how often entrant is brought

TEST:
Basically we will use the standard bootstrap method. 

NOTE:
I tried innovative methods in my last 2 tests and I am heading deadlline, so instead of moving along newer ideas I guess I will use standard methods to have some progress to show.

EXPECTATION:
I have expect to see a few clear levers dominating, maybe lower price or better label something like that, if this happens I get an indication that somehow this lever dominates over all other parameters of the user.

RAN experiments/lever_rates.py

RESULT & CONCLUSION:
The result is stored in result/result_test2.txt
From the results, I see that for price, 3.49 has more entrant purchase rate than 4.49 which indeed has more entrant purchase rate than 5.49, and the observed differences are 10% for 3.49 and 4.49 and 6.1% for 4.49 and 5.49, consistently having 16.1% between 3.49 and 5.49. we see direction of more entrant purchase rate towards lower price, as I expected through intuition. we dont see any clear direction in "message". for channel, we see the direction of more entrant purchase towards online than retail, and the observed differnce is 5.2%. 

MODEL M0:

OBJECTIVE:
To make the most basic model just based on "choice"

HOW?
We will design this model just based on the "choice" column. We dont take in care of any other parameter. We will just find out probabilities of each choice being chosen. Then we use log-loss for scoring. We use cross validation for scoring to get a better score, we divide data in 5 blocks based on household and obtain probability from 4 blocks and compute score on the 5th one, doing so 5 times, so that we get a honest score.

EXPECTATION:
This model is a simple model. But given we have 5 choice (including none), so for the most fair case if I take 20% probability, we should get the score around 1.61. The lower the score from 1.61, the more bias is towards one choice. We check scores based on blocks of household.

I am curious to see if M0 will beat the baseline model on the score. If baseline wins, we get to know that yeah, the assumptions we got from baseline maybe correct, but if M0 wins we conclude that setting hard-coded weights are worse.

WROTE experiments/m0.py
EVALUATED MODEL with experiments/evaluation.py

RESULT & CONCLUSION:
The result is stored in results/results_m0.txt
From the results, I clearly see that baseline model beats M0. So now what I conclude is a very obvious conclusion that the other parameters are important and we need to consider them, as that is considered by baseline model, even though hard-coded. Might be the assumptions baseline made are true, but there is no certainity of this fact. 

MODEL M1:

OBJECTIVE:
To add household traits to the model. Currently our model M0 uses just the "choice", which is beaten by the baseline model, which has considered input parameters, so the first few parameters, i would like to add to the model would be around household traits as I assume what the customer chooses greatly depends upon who the customer is.

HOW?
We use multinomial logistic regression, where in we start with weights as 0, and then adjusted according to slope of the log loss over all training points, until the slope becomes almost 0. The inputs we add are "usual brand", "budget", "income band", "cleaning spend". For evaluation, we use cross validation just as we did in M0

EXPECTATION:
I expect this model to beat the baseline model, but I am not very sure about that because baseline also had weights for repeats and many other input parameters, we havent trained on.