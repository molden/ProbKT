:- use_module(library(lists)).
nn(mnist_net,[X],Y,[0,1,2,3,4,5,6,7,8,9]) :: digit(X,Y).

sum([],0).
sum([X|T],Y):- sum(T,Z), Y is X+Z.

roll([],L,L).
roll([H|T],A,L):- roll(T,[Y|A],L), digit(H,Y).

sum_digits(X,Y) :- roll(X,[],Result), sum(Result,Y).
