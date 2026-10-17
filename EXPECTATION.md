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
