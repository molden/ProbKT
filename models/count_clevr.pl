:- use_module(library(lists)).
nn(mnist_net,[X],Y,[0,1,2,3,4,5,6,7,8,9,10,11]) :: digit(X,Y).

count([],X,0).
count([X|T],X,Y):- count(T,X,Z), Y is 1+Z.
count([X1|T],X,Z):- X1\=X,count(T,X,Z).

countall(List,X,C) :-
    sort(List,List1),
    member(X,List1),
    count(List,X,C).

roll([],L,L).
roll([H|T],A,L):- roll(T,[Y|A],L), digit(H,Y).

countpart(List,[],[]).
countpart(List,[H|T],[F|L]):- countall(List,H,F), countpart(List,T,L).

count_objects(X,L,C) :- roll(X,[],Result), countpart(Result,L,C).
