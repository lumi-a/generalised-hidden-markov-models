
## Exercises

1. Using your programming language of choice, implement a function that takes in:  
   1. The transition matrices of a GHMM (a three-tensor)  
   2. The initial vector   
   3. A sequence of tokens

	and outputs the probability of observing that particular sequence of tokens. 

2. Test your function on the random-random-XOR (RRXOR) process when initialised in the state η(∅) \= \[1 0 0 0 0\]. Make sure you assign zero probability to XOR violations.    
3. Convince yourself that that the definition of a GHMM allows one to interpret η(∅)T(w1)T(w2)...T(wn)ϕ as the probability of emitting the sequence w1 w2 … wn.  