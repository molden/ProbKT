:- use_module(library(lists)).
nn(mnist_net,[X],Y,[0,1,2,3,4,5,6,7,8,9]) :: digit(X,Y).

prod([],1).
prod([X|T],Y):- prod(T,Z), Y is X*Z.

roll([],L,L).
roll([H|T],A,L):- roll(T,[Y|A],L), digit(H,Y).

prod_digits(X,Y) :- roll(X,[],Result), prod(Result,Y).
