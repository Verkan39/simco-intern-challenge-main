The first thing, I am specifically interested in knowing whether the three parameters, price, channel and shelf-message are independent? My intiution isnt sure about this. I will try a test for finding this out.

How will I test this?
I will take a particular label(s), example, entrant_price=3.49, or entrant_price=4.49, channel=online, something like this.
Now for this label(s), I see when these labels were given how many of them converted to sales. Now the question is wont we miss the repeat orders?
So to cater the repeat problem, we can tweak this algo a little. What I do is I count the number of times entrant was chosen by that household after the first time the given label(s) convert it to sales.
Then I check independence with classical probability rules for pairs and then in trios. [ P(A∩B) = P(A)·P(B) ]  

My concern :
The data size is small, there is a possibility that we conclude noise as a relation. 

If noise is seen as relation, and we cannot conclude independence,I will have to change my mind from finding answer of relation/indepence between them.