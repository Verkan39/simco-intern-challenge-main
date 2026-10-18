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

RAN label_probabilities.py

RESULT & CONCLUSION :
The result is stored in result_test1.txt.
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