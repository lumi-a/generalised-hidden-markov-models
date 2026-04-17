
## Exercises

**Core exercises.**

1. Using your programming language of choice, implement a function that takes in:  
   1. The transition matrices of a GHMM (a three-tensor)  
   2. The initial vector   
   3. A sequence of tokens

	and outputs the probability of observing that particular sequence of tokens. 

2. Test your function on the random-random-XOR (RRXOR) process when initialised in the state η(∅) \= \[1 0 0 0 0\]. Make sure you assign zero probability to XOR violations.    
3. Convince yourself that that the definition of a GHMM allows one to interpret η(∅)T(w1)T(w2)...T(wn)ϕ as the probability of emitting the sequence w1 w2 … wn.  
4. Derive the transition matrices for the zero-random-random process. 

**Extension exercises.** 

1. The RRXOR process can be viewed as a particular instance of a more general process called *random-random-Mod p* (RRModp) where p \= 2,3,..., and RRXOR \= RRMod2.   
   1. What are the transition matrices for the RRMod3 process?   
   2. What are the transition matrices for the RRModp process?   
   3. Do you notice anything about the matrices sitting in the blocks?  
2. The RRModp process can be similarly viewed as a particular instance of a more general process that computes the product of two *group* elements (for RRModp the cyclic group is the relevant one). Identify another group whose multiplication can be expressed as an HMM.
